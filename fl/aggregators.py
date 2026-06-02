import collections
import copy
import logging
import math
import time
from abc import ABC, abstractmethod

import torch

from fl.bayes_utils import (
    compute_optimal_local_posterior,
    compute_quadratic_meta_terms,
    get_bayes_expert_state,
    get_client_expert_evidence,
    group_expert_keys,
    parse_expert_ref,
)
from utils.utils import get_experiment_stem


# Helpers reserved for the closed_form_weighted Bayes expert aggregation mode.
def _sanitize_precision_tensor(A, args):
    if not torch.is_tensor(A):
        return None
    A = A.detach()
    if not torch.isfinite(A).all().item():
        return None
    precision_min = float(getattr(args, "bayes_weighted_precision_min", 1.0e-8))
    precision_max = float(getattr(args, "bayes_weighted_precision_max", 1.0e8))
    return A.clamp(min=precision_min, max=precision_max)


def _sanitize_var_tensor(V, args):
    if not torch.is_tensor(V):
        return None
    V = V.detach()
    if not torch.isfinite(V).all().item():
        return None
    var_min = float(getattr(args, "bayes_weighted_var_min", 1.0e-8))
    var_max = float(getattr(args, "bayes_weighted_var_max", 1.0e8))
    return V.clamp(min=var_min, max=var_max)


def _summarize_finite_scalars(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _build_shared_precision_calibration(client_payloads, expert_keys, args):
    mode = str(getattr(args, "bayes_weighted_precision_calibration", "none")).lower()
    target = float(getattr(args, "bayes_weighted_precision_target", 100.0))
    eps = max(float(getattr(args, "bayes_weighted_eps", 1.0e-8)), 1.0e-12)
    scales = collections.OrderedDict((key, 1.0) for key in expert_keys)
    raw_medians = []
    invalid_count = 0
    if mode == "median_target":
        for key in expert_keys:
            client_medians = []
            for payload in client_payloads:
                raw_precision = payload.get("precision_state", {}).get(key)
                if not torch.is_tensor(raw_precision):
                    continue
                raw_precision = raw_precision.detach()
                valid_values = raw_precision[torch.isfinite(raw_precision) & (raw_precision > 0)]
                if valid_values.numel() > 0:
                    client_medians.append(float(torch.quantile(valid_values.double(), 0.5).item()))
            if client_medians:
                raw_median = float(torch.quantile(torch.tensor(client_medians, dtype=torch.float64), 0.5).item())
            else:
                raw_median = None
            if raw_median is None or not math.isfinite(raw_median) or raw_median <= 0:
                invalid_count += 1
                continue
            scale = target / (raw_median + eps)
            if not math.isfinite(scale) or scale <= 0:
                invalid_count += 1
                continue
            scales[key] = float(scale)
            raw_medians.append(raw_median)

    scale_stats = _summarize_finite_scalars(scales.values())
    raw_median_stats = _summarize_finite_scalars(raw_medians)
    return scales, {
        "precision_calibration_mode": mode,
        "precision_target": target,
        "precision_scale_mean": scale_stats["mean"],
        "precision_scale_min": scale_stats["min"],
        "precision_scale_max": scale_stats["max"],
        "raw_precision_median_mean": raw_median_stats["mean"],
        "raw_precision_median_min": raw_median_stats["min"],
        "raw_precision_median_max": raw_median_stats["max"],
        "invalid_precision_calibration_count": invalid_count,
    }


def _apply_shared_precision_calibration(
    local_precision_state,
    expert_keys,
    scales,
    args,
    precision_clip_diag=None,
):
    if not isinstance(local_precision_state, dict):
        return local_precision_state
    precision_min = float(getattr(args, "bayes_weighted_precision_min", 1.0e-8))
    precision_max = float(getattr(args, "bayes_weighted_precision_max", 1.0e8))
    effective_state = collections.OrderedDict()
    for key in expert_keys:
        raw_precision = local_precision_state.get(key)
        if not torch.is_tensor(raw_precision) or not torch.isfinite(raw_precision).all().item():
            effective_state[key] = raw_precision
            continue
        scaled_precision = raw_precision.detach().double() * float(scales.get(key, 1.0))
        if precision_clip_diag is not None:
            valid = torch.isfinite(scaled_precision)
            precision_clip_diag["precision_total_count"] += int(valid.sum().item())
            precision_clip_diag["precision_clip_max_count"] += int(
                ((scaled_precision > precision_max) & valid).sum().item()
            )
            precision_clip_diag["precision_clip_min_count"] += int(
                ((scaled_precision < precision_min) & valid).sum().item()
            )
        effective_state[key] = scaled_precision.clamp(min=precision_min, max=precision_max)
    return effective_state


def _compute_local_posterior_for_client(
    local_mean_state,
    local_precision_state,
    prior_mean_state,
    prior_var_state,
    n0,
    args,
):
    states = [local_mean_state, local_precision_state, prior_mean_state, prior_var_state]
    if not all(isinstance(state, dict) for state in states):
        return None

    m_star_state = collections.OrderedDict()
    v_star_state = collections.OrderedDict()
    sanitized_precision_state = collections.OrderedDict()
    with torch.no_grad():
        for key, prior_mean in prior_mean_state.items():
            local_mean = local_mean_state.get(key)
            local_precision = local_precision_state.get(key)
            prior_var = prior_var_state.get(key)
            tensors = [local_mean, local_precision, prior_mean, prior_var]
            if not all(torch.is_tensor(value) for value in tensors):
                return None
            if any(value.shape != prior_mean.shape for value in tensors):
                return None

            prior_mean = prior_mean.detach()
            local_mean = local_mean.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            local_precision = local_precision.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            prior_var = prior_var.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            local_precision = _sanitize_precision_tensor(local_precision, args)
            prior_var = _sanitize_var_tensor(prior_var, args)
            n0_tensor = torch.as_tensor(n0, device=prior_mean.device, dtype=prior_mean.dtype).detach()
            if (
                local_precision is None
                or prior_var is None
                or not torch.isfinite(local_mean).all().item()
                or not torch.isfinite(prior_mean).all().item()
                or not torch.isfinite(n0_tensor).all().item()
                or n0_tensor.numel() != 1
                or n0_tensor.item() < 0
            ):
                return None

            prior_eff_prec = n0_tensor / prior_var
            v_star = 1.0 / (local_precision + prior_eff_prec)
            m_star = v_star * (local_precision * local_mean + prior_eff_prec * prior_mean)
            if not torch.isfinite(v_star).all().item() or not torch.isfinite(m_star).all().item():
                return None
            m_star_state[key] = m_star
            v_star_state[key] = v_star
            sanitized_precision_state[key] = local_precision
    return m_star_state, v_star_state, sanitized_precision_state


def _compute_evidence_score_scalar(
    local_mean_state,
    local_precision_state,
    prior_mean_state,
    prior_var_state,
    m_star_state,
    v_star_state,
    n0,
    args,
):
    states = [
        local_mean_state,
        local_precision_state,
        prior_mean_state,
        prior_var_state,
        m_star_state,
        v_star_state,
    ]
    if not all(isinstance(state, dict) for state in states):
        return None

    f_terms = []
    g_terms = []
    eps = float(getattr(args, "bayes_weighted_eps", 1.0e-8))
    with torch.no_grad():
        for key, prior_mean in prior_mean_state.items():
            local_mean = local_mean_state.get(key)
            local_precision = local_precision_state.get(key)
            prior_var = prior_var_state.get(key)
            m_star = m_star_state.get(key)
            v_star = v_star_state.get(key)
            tensors = [local_mean, local_precision, prior_mean, prior_var, m_star, v_star]
            if not all(torch.is_tensor(value) for value in tensors):
                return None
            if any(value.shape != prior_mean.shape for value in tensors):
                return None

            prior_mean = prior_mean.detach()
            local_mean = local_mean.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            local_precision = local_precision.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            prior_var = prior_var.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            m_star = m_star.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            v_star = v_star.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
            local_precision = _sanitize_precision_tensor(local_precision, args)
            prior_var = _sanitize_var_tensor(prior_var, args)
            v_star = _sanitize_var_tensor(v_star, args)
            n0_tensor = torch.as_tensor(n0, device=prior_mean.device, dtype=prior_mean.dtype).detach()
            if (
                local_precision is None
                or prior_var is None
                or v_star is None
                or not torch.isfinite(local_mean).all().item()
                or not torch.isfinite(prior_mean).all().item()
                or not torch.isfinite(m_star).all().item()
                or not torch.isfinite(n0_tensor).all().item()
                or n0_tensor.numel() != 1
                or n0_tensor.item() < 0
            ):
                return None

            f_terms.append((local_precision * v_star + local_precision * (m_star - local_mean).square()).mean())
            g_terms.append(
                (
                    torch.log(prior_var + eps)
                    - torch.log(v_star + eps)
                    + n0_tensor * v_star / (prior_var + eps)
                    + n0_tensor * (m_star - prior_mean).square() / (prior_var + eps)
                ).mean()
            )
        if len(f_terms) == 0:
            return None
        score = 0.5 * torch.stack(f_terms).mean() + 0.5 * torch.stack(g_terms).mean()
        if not torch.isfinite(score).item():
            return None
        return float(score.item())


def _scores_to_beta(scores, args):
    min_valid_clients = int(getattr(args, "bayes_weighted_min_valid_clients", 2))
    if len(scores) < min_valid_clients:
        return None

    score_values = []
    for score in scores:
        if torch.is_tensor(score):
            if score.numel() != 1:
                return None
            score = score.detach().cpu().item()
        try:
            score_values.append(float(score))
        except (TypeError, ValueError):
            return None
    score_tensor = torch.tensor(score_values, dtype=torch.float64)
    if not torch.isfinite(score_tensor).all().item():
        return None

    eps = float(getattr(args, "bayes_weighted_eps", 1.0e-8))
    sigma = score_tensor.std(unbiased=False)
    if sigma.item() < eps:
        return [1.0 / len(scores)] * len(scores)
    z = (score_tensor - score_tensor.mean()) / (sigma + eps)
    score_clip = float(getattr(args, "bayes_weighted_score_clip", 3.0))
    score_tau = float(getattr(args, "bayes_weighted_score_tau", 0.5))
    beta = torch.softmax(-score_tau * z.clamp(min=-score_clip, max=score_clip), dim=0)
    return beta.tolist()


class Aggregator(ABC):
    # 聚合器统一接口。后续新增聚合方法时，只需要新增实现类并在 build_aggregator 中注册。
    @abstractmethod
    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        pass


class FedAvgAggregator(Aggregator):
    # 标准 FedAvg：
    # 对完整 state_dict 做按客户端样本数加权平均，权重 w_i = n_i / sum_j n_j。
    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        if len(client_updates) == 0:
            raise ValueError("FedAvg requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        total_weight = sum(client_weights)
        if total_weight <= 0:
            raise ValueError("FedAvg requires positive total client weight")

        aggregated_state = collections.OrderedDict()
        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()
            if torch.is_floating_point(first_value):
                aggregated_state[key] = torch.zeros_like(first_value)
                for update, weight in zip(client_updates, client_weights):
                    aggregated_state[key] += update[key].detach().cpu() * (weight / total_weight)
            else:
                # 非浮点 buffer 通常不能加权平均，沿用第一个客户端的值。
                aggregated_state[key] = first_value.clone()

        return aggregated_state


class ClientAvgAggregator(Aggregator):
    # 对完整 state_dict 做客户端等权平均，每个客户端权重均为 1 / num_clients。
    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        if len(client_updates) == 0:
            raise ValueError("ClientAvg requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        num_clients = len(client_updates)
        avg_weight = 1.0 / num_clients
        aggregated_state = collections.OrderedDict()
        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()
            if torch.is_floating_point(first_value):
                aggregated_state[key] = torch.zeros_like(first_value)
                for update in client_updates:
                    aggregated_state[key] += update[key].detach().cpu() * avg_weight
            else:
                # 非浮点 buffer 通常不能加权平均，沿用第一个客户端的值。
                aggregated_state[key] = first_value.clone()

        return aggregated_state


class ExpertFedAvgAggregator(Aggregator):
    # FL + MoE 专家级 FedAvg：
    # - 普通共享层仍按客户端训练样本数 n_i 做标准 FedAvg；
    # - blocks.{layer}.ffn.experts.{expert_id}.* 参数按该层该专家实际处理的 token 数 n_{i,l,e} 加权。
    def __init__(self):
        pass

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        if len(client_updates) == 0:
            raise ValueError("ExpertFedAvg requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        expert_weights = kwargs.get("expert_weights")
        if expert_weights is None:
            raise ValueError("ExpertFedAvg requires expert_weights for expert-level aggregation")
        if len(expert_weights) != len(client_updates):
            raise ValueError("expert_weights and client_updates must have the same length")

        global_state = global_model.state_dict() if global_model is not None else None
        aggregated_state = collections.OrderedDict()
        total_client_weight = sum(client_weights)
        if total_client_weight <= 0:
            raise ValueError("ExpertFedAvg requires positive total client weight")

        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.clone()
                continue

            expert_ref = self._parse_expert_ref(key)
            if expert_ref is None:
                weights = client_weights
            else:
                layer_id, expert_id = expert_ref
                weights = [
                    self._get_expert_weight(client_usage, layer_id, expert_id)
                    for client_usage in expert_weights
                ]

            total_weight = sum(weights)
            if total_weight <= 0:
                # 某一轮没有客户端使用该专家时，不用随机客户端覆盖它，保留服务端旧参数更稳。
                if global_state is not None:
                    aggregated_state[key] = global_state[key].detach().cpu().clone()
                else:
                    aggregated_state[key] = first_value.clone()
                continue

            aggregated_state[key] = torch.zeros_like(first_value)
            for update, weight in zip(client_updates, weights):
                aggregated_state[key] += update[key].detach().cpu() * (weight / total_weight)

        return aggregated_state

    def _parse_expert_ref(self, key):
        expert_ref = parse_expert_ref(key)
        if expert_ref is None:
            return None
        layer_id, expert_id = expert_ref
        return layer_id, int(expert_id)

    def _get_expert_weight(self, client_usage, layer_id, expert_id):
        if isinstance(client_usage, dict):
            if layer_id is None:
                usage = client_usage.get("expert_activations")
            else:
                layer_stats = client_usage.get("expert_stats_by_layer", {}).get(str(layer_id), {})
                usage = layer_stats.get("expert_activations")
                if usage is None:
                    usage = client_usage.get("expert_activations_by_layer", {}).get(str(layer_id))

            if usage is None:
                return 0.0
            if expert_id >= len(usage):
                raise ValueError(f"Missing expert weight for expert id {expert_id}")
            return float(usage[expert_id])

        if expert_id >= len(client_usage):
            raise ValueError(f"Missing expert weight for expert id {expert_id}")
        return float(client_usage[expert_id])


class ExpertBayesMetaAggregator(Aggregator):
    # Shared parameters use FedAvg. Experts with local evidence use the basic Bayes meta update.
    def __init__(self, args):
        self.args = args
        self.base_fedavg = FedAvgAggregator()
        self.min_precision = 1.0e-6
        self.max_precision = 1.0e6
        self.max_n0 = 1.0e6
        self.gamma0_init = max(float(getattr(args, "bayes_gamma0_init", 1.0)), self.min_precision)
        self.n0_init = max(float(getattr(args, "bayes_n0_init", 1.0)), self.min_precision)
        self.meta_steps = max(int(getattr(args, "bayes_meta_steps", 5)), 1)
        self.meta_lr = float(getattr(args, "bayes_meta_lr", 0.001))
        self.meta_update_mode = str(getattr(args, "bayes_meta_update_mode", "optimizer")).lower()
        self.weighted_score_tau = float(getattr(args, "bayes_weighted_score_tau", 0.5))
        self.weighted_score_clip = float(getattr(args, "bayes_weighted_score_clip", 3.0))
        self.weighted_var_rho = float(getattr(args, "bayes_weighted_var_rho", 0.05))
        self.weighted_eps = float(getattr(args, "bayes_weighted_eps", 1.0e-8))
        self.weighted_min_valid_clients = int(getattr(args, "bayes_weighted_min_valid_clients", 2))
        self.weighted_precision_min = float(getattr(args, "bayes_weighted_precision_min", 1.0e-8))
        self.weighted_precision_max = float(getattr(args, "bayes_weighted_precision_max", 1.0e8))
        self.weighted_precision_calibration = str(
            getattr(args, "bayes_weighted_precision_calibration", "none")
        ).lower()
        self.weighted_precision_target = float(getattr(args, "bayes_weighted_precision_target", 100.0))
        self.weighted_var_min = float(getattr(args, "bayes_weighted_var_min", 1.0e-8))
        self.weighted_var_max = float(getattr(args, "bayes_weighted_var_max", 1.0e8))
        self.weighted_diag = bool(
            getattr(args, "bayes_weighted_diag", False) or getattr(args, "bayes_direction_diag", False)
        )
        if self.meta_update_mode not in {"optimizer", "closed_form_weighted"}:
            raise ValueError(
                "bayes_meta_update_mode must be one of: optimizer, closed_form_weighted"
            )
        if self.weighted_precision_calibration not in {"none", "median_target"}:
            raise ValueError(
                "bayes_weighted_precision_calibration must be one of: none, median_target"
            )
        self.update_precision = bool(getattr(args, "bayes_update_precision", True))
        self.update_strength = bool(getattr(args, "bayes_update_strength", True))
        self.meta_device = self._resolve_meta_device(args)
        self.empty_cache_after_aggregation = bool(
            getattr(args, "bayes_empty_cache_after_aggregation", False)
        )
        precision_source = str(getattr(args, "bayes_precision_source", "sgld_variance")).lower()
        sgld_fit_mode = str(getattr(args, "bayes_sgld_fit_mode", "adam_noise")).lower()
        precision_mode = str(getattr(args, "bayes_precision_mode", "floor_inverse")).lower()
        if precision_source != "sgld_variance":
            raise ValueError("bayes_precision_source now only supports: sgld_variance")
        if sgld_fit_mode not in {"adam_noise", "sgd_noise"}:
            raise ValueError("bayes_sgld_fit_mode must be one of: adam_noise, sgd_noise")
        if precision_mode != "floor_inverse":
            raise ValueError("bayes_precision_mode now only supports: floor_inverse")
        print(
            "[ExpertBayesMetaAggregator] "
            "bayes_precision_source=sgld_variance "
            f"bayes_sgld_noise_mode={sgld_fit_mode} "
            "bayes_precision_method=floor_inverse "
            f"bayes_meta_device={self.meta_device}"
        )

    def _resolve_meta_device(self, args):
        requested = str(getattr(args, "bayes_meta_device", "auto")).lower()
        base_device = str(getattr(args, "device", "cpu"))
        if requested == "auto":
            if base_device.startswith("cuda") and torch.cuda.is_available():
                return torch.device(base_device)
            return torch.device("cpu")
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                print(
                    "[ExpertBayesMetaAggregator] CUDA requested for "
                    "bayes_meta_device but not available; fallback to CPU"
                )
                return torch.device("cpu")
            return torch.device(requested)
        if requested == "cpu":
            return torch.device("cpu")
        raise ValueError(f"Unsupported bayes_meta_device: {requested}")

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        aggregate_start_time = time.perf_counter()
        if len(client_updates) == 0:
            raise ValueError("ExpertBayesMeta requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        expert_evidence = kwargs.get("expert_evidence")
        if expert_evidence is None:
            raise ValueError("ExpertBayesMeta requires expert_evidence from clients")
        if len(expert_evidence) != len(client_updates):
            raise ValueError("expert_evidence and client_updates must have the same length")
        bayes_state = kwargs.get("bayes_state")
        if bayes_state is None:
            raise ValueError("ExpertBayesMeta requires bayes_state from server")

        global_state = global_model.state_dict() if global_model is not None else client_updates[0]
        aggregated_state = self.base_fedavg.aggregate(
            client_updates=client_updates,
            client_weights=client_weights,
            global_model=global_model,
        )
        updated_bayes_state = copy.deepcopy(bayes_state)
        expert_groups = group_expert_keys(global_state)
        metrics = {
            "updated_experts": 0,
            "skipped_experts": 0,
            "evidence_clients": 0,
            "expert_param_groups": len(expert_groups),
            "local_posteriors": 0,
            "bayes_meta_device": str(self.meta_device),
            "expert_meta_stats": {},
        }

        for (layer_id, expert_id), expert_keys in expert_groups.items():
            expert_params, contributing_clients, local_posterior_count, expert_metric = self._aggregate_expert_group(
                layer_id=layer_id,
                expert_id=expert_id,
                expert_keys=expert_keys,
                global_state=global_state,
                updated_bayes_state=updated_bayes_state,
                expert_evidence=expert_evidence,
            )
            metrics["expert_meta_stats"][f"{layer_id}.{expert_id}"] = expert_metric
            if contributing_clients > 0:
                metrics["updated_experts"] += 1
                metrics["evidence_clients"] += contributing_clients
                metrics["local_posteriors"] += local_posterior_count
            else:
                metrics["skipped_experts"] += 1
            for key, value in expert_params.items():
                aggregated_state[key] = value

        if self.meta_update_mode == "closed_form_weighted":
            metrics["bayes_weighted_updated_experts"] = metrics["updated_experts"]
            metrics["bayes_weighted_skipped_experts"] = metrics["skipped_experts"]
        updated_bayes_state["round"] = int(updated_bayes_state.get("round", 0)) + 1
        metrics["bayes_aggregation_time_sec"] = round(time.perf_counter() - aggregate_start_time, 4)
        if self.meta_device.type == "cuda" and self.empty_cache_after_aggregation:
            torch.cuda.empty_cache()
        return {
            "model_state": aggregated_state,
            "bayes_state": updated_bayes_state,
            "metrics": metrics,
        }

    def _aggregate_expert_group(
        self,
        layer_id,
        expert_id,
        expert_keys,
        global_state,
        updated_bayes_state,
        expert_evidence,
    ):
        expert_start_time = time.perf_counter()
        prior_state = self._get_or_init_prior_state(
            updated_bayes_state=updated_bayes_state,
            layer_id=layer_id,
            expert_id=expert_id,
            expert_keys=expert_keys,
            global_state=global_state,
        )
        client_payloads = self._collect_client_payloads(expert_evidence, layer_id, expert_id, expert_keys)
        if len(client_payloads) == 0:
            expert_metric = self._build_expert_metric(
                layer_id=layer_id,
                expert_id=expert_id,
                prior_state=prior_state,
                meta_loss=None,
                contributing_clients=0,
                local_posterior_count=0,
                status="skipped",
                expert_keys=expert_keys,
                client_payloads=client_payloads,
            )
            expert_metric["expert_meta_time_sec"] = round(time.perf_counter() - expert_start_time, 4)
            return {
                key: global_state[key].detach().cpu().clone()
                for key in expert_keys
            }, 0, 0, expert_metric

        prior_n0 = self._get_prior_n0(prior_state)
        if self.meta_update_mode == "closed_form_weighted":
            return self._aggregate_expert_group_closed_form_weighted(
                expert_keys=expert_keys,
                global_state=global_state,
                prior_state=prior_state,
                prior_n0=prior_n0,
                client_payloads=client_payloads,
                layer_id=layer_id,
                expert_id=expert_id,
                expert_start_time=expert_start_time,
            )
        optimized_mean_state, optimized_log_precision_state, optimized_log_n0, local_posterior_count, meta_loss = self._optimize_expert_prior(
            expert_keys=expert_keys,
            global_state=global_state,
            prior_state=prior_state,
            prior_n0=prior_n0,
            client_payloads=client_payloads,
            layer_id=layer_id,
            expert_id=expert_id,
        )
        aggregated_params = {
            key: optimized_mean_state[key].detach().cpu().to(dtype=global_state[key].dtype).clone()
            for key in expert_keys
        }
        if self.update_precision:
            for key in expert_keys:
                prior_state["log_precision_state"][key] = optimized_log_precision_state[key].detach().cpu()
        if self.update_strength:
            prior_state["log_n0"] = optimized_log_n0.detach().cpu()

        expert_metric = self._build_expert_metric(
            layer_id=layer_id,
            expert_id=expert_id,
            prior_state=prior_state,
            meta_loss=meta_loss,
            contributing_clients=len(client_payloads),
            local_posterior_count=local_posterior_count,
            status="updated",
            optimized_log_precision_state=optimized_log_precision_state,
            optimized_log_n0=optimized_log_n0,
            expert_keys=expert_keys,
            client_payloads=client_payloads,
            global_state=global_state,
            optimized_mean_state=optimized_mean_state,
        )
        expert_metric["expert_meta_time_sec"] = round(time.perf_counter() - expert_start_time, 4)
        return aggregated_params, len(client_payloads), local_posterior_count, expert_metric

    def _aggregate_expert_group_closed_form_weighted(
        self,
        expert_keys,
        global_state,
        prior_state,
        prior_n0,
        client_payloads,
        layer_id,
        expert_id,
        expert_start_time,
    ):
        def build_skipped_result(valid_client_payloads):
            expert_metric = self._build_expert_metric(
                layer_id=layer_id,
                expert_id=expert_id,
                prior_state=prior_state,
                meta_loss=None,
                contributing_clients=0,
                local_posterior_count=0,
                status="skipped",
                expert_keys=expert_keys,
                client_payloads=valid_client_payloads,
            )
            expert_metric["expert_meta_time_sec"] = round(time.perf_counter() - expert_start_time, 4)
            return {
                key: global_state[key].detach().cpu().clone()
                for key in expert_keys
            }, 0, 0, expert_metric

        prior_n0 = prior_n0.detach().to(device=self.meta_device, dtype=torch.float32)
        if not torch.isfinite(prior_n0).all().item() or prior_n0.numel() != 1 or prior_n0.item() <= 0:
            return build_skipped_result([])

        prior_mean_state = collections.OrderedDict()
        prior_var_state = collections.OrderedDict()
        for key in expert_keys:
            prior_mean = global_state[key].detach().to(device=self.meta_device, dtype=torch.float32).clone()
            log_precision = prior_state.get("log_precision_state", {}).get(key)
            if (
                not torch.is_floating_point(prior_mean)
                or not torch.is_tensor(log_precision)
                or not torch.is_floating_point(log_precision)
                or log_precision.shape != prior_mean.shape
            ):
                return build_skipped_result([])
            log_precision = log_precision.detach().to(device=self.meta_device, dtype=torch.float32)
            prior_precision = _sanitize_precision_tensor(torch.exp(log_precision), self.args)
            if prior_precision is None:
                return build_skipped_result([])
            prior_var = _sanitize_var_tensor(1.0 / prior_precision, self.args)
            if prior_var is None:
                return build_skipped_result([])
            prior_mean_state[key] = prior_mean
            prior_var_state[key] = prior_var

        calibration_payloads = []
        if self.weighted_precision_calibration == "median_target":
            for payload in client_payloads:
                posterior = _compute_local_posterior_for_client(
                    local_mean_state=payload.get("mean_state"),
                    local_precision_state=payload.get("precision_state"),
                    prior_mean_state=prior_mean_state,
                    prior_var_state=prior_var_state,
                    n0=prior_n0,
                    args=self.args,
                )
                if posterior is not None:
                    calibration_payloads.append(payload)
        precision_scales, precision_calibration_diag = _build_shared_precision_calibration(
            client_payloads=calibration_payloads,
            expert_keys=expert_keys,
            args=self.args,
        )

        valid_clients = []
        for payload in client_payloads:
            local_precision_state = payload.get("precision_state")
            precision_clip_diag = None
            if self.weighted_precision_calibration == "median_target":
                if bool(getattr(self.args, "bayes_weighted_diag", False)):
                    precision_clip_diag = {
                        "precision_clip_max_count": 0,
                        "precision_clip_min_count": 0,
                        "precision_total_count": 0,
                    }
                local_precision_state = _apply_shared_precision_calibration(
                    local_precision_state=local_precision_state,
                    expert_keys=expert_keys,
                    scales=precision_scales,
                    args=self.args,
                    precision_clip_diag=precision_clip_diag,
                )
            posterior = _compute_local_posterior_for_client(
                local_mean_state=payload.get("mean_state"),
                local_precision_state=local_precision_state,
                prior_mean_state=prior_mean_state,
                prior_var_state=prior_var_state,
                n0=prior_n0,
                args=self.args,
            )
            if posterior is None:
                continue
            m_star_state, v_star_state, sanitized_precision_state = posterior
            score = _compute_evidence_score_scalar(
                local_mean_state=payload.get("mean_state"),
                local_precision_state=sanitized_precision_state,
                prior_mean_state=prior_mean_state,
                prior_var_state=prior_var_state,
                m_star_state=m_star_state,
                v_star_state=v_star_state,
                n0=prior_n0,
                args=self.args,
            )
            if score is None:
                continue
            valid_clients.append({
                "payload": payload,
                "m_star_state": m_star_state,
                "sanitized_precision_state": sanitized_precision_state,
                "score": score,
                "precision_clip_diag": precision_clip_diag,
            })

        beta = _scores_to_beta([client["score"] for client in valid_clients], self.args)
        if beta is None:
            return build_skipped_result([client["payload"] for client in valid_clients])

        rho = float(self.weighted_var_rho)
        if not math.isfinite(rho):
            return build_skipped_result([client["payload"] for client in valid_clients])

        new_mean_state = collections.OrderedDict()
        new_log_precision_state = collections.OrderedDict()
        with torch.no_grad():
            for key in expert_keys:
                new_mean = torch.zeros_like(prior_mean_state[key])
                a_hat = torch.zeros_like(prior_mean_state[key])
                for client_beta, client in zip(beta, valid_clients):
                    new_mean += client_beta * client["m_star_state"][key]
                    a_hat += client_beta * client["sanitized_precision_state"][key]
                if not torch.isfinite(new_mean).all().item() or not torch.isfinite(a_hat).all().item():
                    return build_skipped_result([client["payload"] for client in valid_clients])

                p_old = prior_n0 / prior_var_state[key]
                p_new = (1.0 - rho) * p_old + rho * a_hat
                if not torch.isfinite(p_new).all().item() or (p_new <= 0).any().item():
                    return build_skipped_result([client["payload"] for client in valid_clients])
                v_new = _sanitize_var_tensor(prior_n0 / p_new, self.args)
                if v_new is None:
                    return build_skipped_result([client["payload"] for client in valid_clients])
                precision_new = _sanitize_precision_tensor(1.0 / v_new, self.args)
                if precision_new is None:
                    return build_skipped_result([client["payload"] for client in valid_clients])
                new_mean_state[key] = new_mean
                new_log_precision_state[key] = torch.log(precision_new)

        aggregated_params = {
            key: new_mean_state[key].detach().cpu().to(dtype=global_state[key].dtype).clone()
            for key in expert_keys
        }
        for key in expert_keys:
            prior_state["log_precision_state"][key] = new_log_precision_state[key].detach().cpu()

        local_posterior_count = len(valid_clients) * len(expert_keys)
        expert_metric = self._build_expert_metric(
            layer_id=layer_id,
            expert_id=expert_id,
            prior_state=prior_state,
            meta_loss=None,
            contributing_clients=len(valid_clients),
            local_posterior_count=local_posterior_count,
            status="updated",
            optimized_log_precision_state=new_log_precision_state,
            optimized_log_n0=prior_state.get("log_n0"),
            expert_keys=expert_keys,
            client_payloads=[client["payload"] for client in valid_clients],
            global_state=global_state,
            optimized_mean_state=new_mean_state,
        )
        if self.weighted_diag:
            expert_metric.update(self._build_closed_form_weighted_diag(
                beta=beta,
                valid_clients=valid_clients,
                prior_var_state=prior_var_state,
                prior_mean_state=prior_mean_state,
                new_mean_state=new_mean_state,
                precision_calibration_diag=precision_calibration_diag,
            ))
        expert_metric["expert_meta_time_sec"] = round(time.perf_counter() - expert_start_time, 4)
        return aggregated_params, len(valid_clients), local_posterior_count, expert_metric

    def _build_closed_form_weighted_diag(
        self,
        beta,
        valid_clients,
        prior_var_state,
        prior_mean_state,
        new_mean_state,
        precision_calibration_diag,
    ):
        def summarize_tensors(tensors, include_std):
            value_sum = 0.0
            value_square_sum = 0.0
            value_count = 0
            value_min = None
            value_max = None
            for value in tensors:
                value = value.detach().double()
                value_sum += float(value.sum().item())
                value_square_sum += float(value.square().sum().item())
                value_count += value.numel()
                tensor_min = float(value.min().item())
                tensor_max = float(value.max().item())
                value_min = tensor_min if value_min is None else min(value_min, tensor_min)
                value_max = tensor_max if value_max is None else max(value_max, tensor_max)
            mean = value_sum / value_count
            summary = {"mean": mean, "min": value_min, "max": value_max}
            if include_std:
                variance = max(value_square_sum / value_count - mean * mean, 0.0)
                summary["std"] = math.sqrt(variance)
            return summary

        beta_tensor = torch.tensor(beta, dtype=torch.float64)
        score_tensor = torch.tensor([client["score"] for client in valid_clients], dtype=torch.float64)
        beta_entropy = float(
            -(beta_tensor * torch.log(beta_tensor.clamp(min=self.weighted_eps))).sum().item()
        )
        precision_stats = summarize_tensors(
            [
                client["sanitized_precision_state"][key]
                for client in valid_clients
                for key in client["sanitized_precision_state"]
            ],
            include_std=True,
        )
        precision_clip_diags = [
            client["precision_clip_diag"]
            for client in valid_clients
            if client["precision_clip_diag"] is not None
        ]
        precision_clip_stats = {}
        if precision_clip_diags:
            precision_clip_max_count = sum(
                diag["precision_clip_max_count"] for diag in precision_clip_diags
            )
            precision_clip_min_count = sum(
                diag["precision_clip_min_count"] for diag in precision_clip_diags
            )
            precision_total_count = sum(
                diag["precision_total_count"] for diag in precision_clip_diags
            )
            precision_clip_count = precision_clip_max_count + precision_clip_min_count
            if precision_total_count > 0:
                precision_clip_max_frac = precision_clip_max_count / precision_total_count
                precision_clip_min_frac = precision_clip_min_count / precision_total_count
                precision_clip_frac = precision_clip_count / precision_total_count
            else:
                precision_clip_max_frac = 0.0
                precision_clip_min_frac = 0.0
                precision_clip_frac = 0.0
            precision_clip_stats = {
                "precision_clip_frac": precision_clip_frac,
                "precision_clip_max_frac": precision_clip_max_frac,
                "precision_clip_min_frac": precision_clip_min_frac,
                "precision_clip_count": precision_clip_count,
                "precision_total_count": precision_total_count,
            }
        prior_var_stats = summarize_tensors(prior_var_state.values(), include_std=False)
        mean_update_square_sum = sum(
            float((new_mean_state[key] - prior_mean_state[key]).double().square().sum().item())
            for key in new_mean_state
        )
        return {
            "beta_min": float(beta_tensor.min().item()),
            "beta_max": float(beta_tensor.max().item()),
            "beta_mean": float(beta_tensor.mean().item()),
            "beta_entropy": beta_entropy,
            "beta_num_clients": len(beta),
            "score_min": float(score_tensor.min().item()),
            "score_max": float(score_tensor.max().item()),
            "score_mean": float(score_tensor.mean().item()),
            "score_std": float(score_tensor.std(unbiased=False).item()),
            "precision_mean": precision_stats["mean"],
            "precision_std": precision_stats["std"],
            "precision_min": precision_stats["min"],
            "precision_max": precision_stats["max"],
            "prior_var_mean": prior_var_stats["mean"],
            "prior_var_min": prior_var_stats["min"],
            "prior_var_max": prior_var_stats["max"],
            "mean_update_norm": math.sqrt(mean_update_square_sum),
            **precision_clip_stats,
            **precision_calibration_diag,
        }

    def _optimize_expert_prior(
        self,
        expert_keys,
        global_state,
        prior_state,
        prior_n0,
        client_payloads,
        layer_id,
        expert_id,
    ):
        prior_mean_params = collections.OrderedDict()
        log_precision_params = collections.OrderedDict()
        optim_params = []
        for key in expert_keys:
            reference_tensor = global_state[key].detach().to(device=self.meta_device, dtype=torch.float32).clone()
            mean_param = torch.nn.Parameter(reference_tensor.clone())
            prior_mean_params[key] = mean_param
            optim_params.append(mean_param)

            log_precision = prior_state.get("log_precision_state", {}).get(key)
            if log_precision is None or not torch.is_floating_point(log_precision):
                log_precision = torch.full_like(reference_tensor, fill_value=math.log(self.gamma0_init))
            else:
                log_precision = log_precision.detach().to(device=self.meta_device, dtype=torch.float32).clone()
            log_precision_param = torch.nn.Parameter(log_precision, requires_grad=self.update_precision)
            log_precision_params[key] = log_precision_param
            if self.update_precision:
                optim_params.append(log_precision_param)

        log_n0_value = prior_state.get("log_n0")
        if log_n0_value is None:
            log_n0_value = torch.tensor(math.log(max(float(prior_n0.item()), self.min_precision)))
        log_n0_value = log_n0_value.detach().to(device=self.meta_device, dtype=torch.float32).clone()
        log_n0_param = torch.nn.Parameter(log_n0_value, requires_grad=self.update_strength)
        if self.update_strength:
            optim_params.append(log_n0_param)

        optimizer = torch.optim.Adam(optim_params, lr=self.meta_lr)
        last_finite_loss = None
        local_posterior_count = 0
        for _ in range(self.meta_steps):
            optimizer.zero_grad()
            meta_loss, local_posterior_count = self._compute_expert_meta_loss(
                expert_keys=expert_keys,
                prior_mean_params=prior_mean_params,
                log_precision_params=log_precision_params,
                log_n0_param=log_n0_param,
                client_payloads=client_payloads,
            )
            if not torch.isfinite(meta_loss).item():
                print(
                    "[ExpertBayesMetaAggregator] warning: non-finite meta_loss "
                    f"layer={layer_id} expert={expert_id}; keeping last finite parameters"
                )
                break
            meta_loss.backward()
            optimizer.step()
            self._project_meta_params(log_precision_params, log_n0_param)
            last_finite_loss = float(meta_loss.detach().cpu().item())

        return prior_mean_params, log_precision_params, log_n0_param, local_posterior_count, last_finite_loss

    def _compute_expert_meta_loss(
        self,
        expert_keys,
        prior_mean_params,
        log_precision_params,
        log_n0_param,
        client_payloads,
    ):
        zero = next(iter(prior_mean_params.values())).new_tensor(0.0)
        client_losses = []
        local_posterior_count = 0
        prior_n0 = torch.exp(log_n0_param).clamp(min=self.min_precision, max=self.max_n0)
        for payload in client_payloads:
            client_loss = zero
            has_local_terms = False
            for key in expert_keys:
                local_mean = payload["mean_state"].get(key)
                local_precision = payload["precision_state"].get(key)
                if local_mean is None or local_precision is None:
                    continue
                prior_mean = prior_mean_params[key]
                local_mean = local_mean.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
                local_precision = local_precision.detach().to(device=prior_mean.device, dtype=prior_mean.dtype)
                prior_precision = torch.exp(log_precision_params[key]).clamp(
                    min=self.min_precision,
                    max=self.max_precision,
                )
                posterior_mean, _, posterior_precision = compute_optimal_local_posterior(
                    local_mean=local_mean,
                    local_precision=local_precision,
                    prior_mean=prior_mean,
                    prior_precision=prior_precision,
                    prior_n0=prior_n0,
                    min_precision=self.min_precision,
                )
                fit_term, regularizer = compute_quadratic_meta_terms(
                    local_mean=local_mean,
                    local_precision=local_precision,
                    prior_mean=prior_mean,
                    prior_precision=prior_precision,
                    prior_n0=prior_n0,
                    posterior_mean=posterior_mean,
                    posterior_precision=posterior_precision,
                    min_precision=self.min_precision,
                )
                client_loss = client_loss + fit_term + 0.5 * regularizer
                has_local_terms = True
                local_posterior_count += 1
            if has_local_terms:
                client_losses.append(client_loss)
        if len(client_losses) == 0:
            return zero, 0
        return torch.stack(client_losses).mean(), local_posterior_count

    def _project_meta_params(self, log_precision_params, log_n0_param):
        log_min_precision = math.log(self.min_precision)
        log_max_precision = math.log(self.max_precision)
        log_max_n0 = math.log(self.max_n0)
        with torch.no_grad():
            for log_precision_param in log_precision_params.values():
                log_precision_param.clamp_(min=log_min_precision, max=log_max_precision)
            log_n0_param.clamp_(min=log_min_precision, max=log_max_n0)

    def _summarize_client_payloads(self, client_payloads, expert_keys):
        precision_values = []
        for payload in client_payloads:
            for key in expert_keys:
                value = payload["precision_state"].get(key)
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    precision_values.append(value.detach().cpu().float().reshape(-1))
        summary = {
            "num_batches_total": sum(int(payload.get("num_batches", 0)) for payload in client_payloads),
            "local_precision_mean": None,
            "local_precision_min": None,
            "local_precision_max": None,
        }
        if precision_values:
            precision_vector = torch.cat(precision_values)
            summary.update({
                "local_precision_mean": round(float(precision_vector.mean().item()), 6),
                "local_precision_min": round(float(precision_vector.min().item()), 6),
                "local_precision_max": round(float(precision_vector.max().item()), 6),
            })
        return summary

    def _summarize_mean_update(self, global_state, optimized_mean_state, expert_keys):
        if global_state is None or optimized_mean_state is None:
            return {"param_delta_rel": None}
        delta_sq = 0.0
        prior_sq = 0.0
        for key in expert_keys:
            prior_value = global_state[key].detach().cpu().float()
            updated_value = optimized_mean_state[key].detach().cpu().float()
            delta_sq += float((updated_value - prior_value).square().sum().item())
            prior_sq += float(prior_value.square().sum().item())
        return {"param_delta_rel": round(math.sqrt(max(delta_sq, 0.0)) / max(math.sqrt(max(prior_sq, 0.0)), 1.0e-12), 6)}

    def _build_expert_metric(
        self,
        layer_id,
        expert_id,
        prior_state,
        meta_loss,
        contributing_clients,
        local_posterior_count,
        status,
        optimized_log_precision_state=None,
        optimized_log_n0=None,
        expert_keys=None,
        client_payloads=None,
        global_state=None,
        optimized_mean_state=None,
    ):
        log_precision_state = optimized_log_precision_state or prior_state.get("log_precision_state", {})
        gamma_values = [
            torch.exp(value.detach().cpu().float()).reshape(-1)
            for value in log_precision_state.values()
            if torch.is_tensor(value) and torch.is_floating_point(value)
        ]
        gamma_vector = torch.cat(gamma_values) if gamma_values else torch.tensor([self.gamma0_init])
        log_n0 = optimized_log_n0 if optimized_log_n0 is not None else prior_state.get("log_n0")
        n0 = self.n0_init if log_n0 is None else float(torch.exp(log_n0.detach().cpu().float()).item())
        expert_keys = expert_keys or []
        client_payloads = client_payloads or []
        metric = {
            "status": status,
            "layer_id": str(layer_id),
            "expert_id": str(expert_id),
            "clients": int(contributing_clients),
            "local_posteriors": int(local_posterior_count),
            "meta_loss": None if meta_loss is None else round(float(meta_loss), 6),
            "n0": round(float(n0), 6),
            "avg_gamma0": round(float(gamma_vector.mean().item()), 6),
        }
        metric.update(self._summarize_client_payloads(client_payloads, expert_keys))
        metric.update(self._summarize_mean_update(global_state, optimized_mean_state, expert_keys))
        return metric

    def _collect_client_payloads(self, expert_evidence, layer_id, expert_id, expert_keys):
        payloads = []
        for client_evidence in expert_evidence:
            evidence = get_client_expert_evidence(client_evidence, layer_id, expert_id)
            if evidence is None:
                continue
            mean_state = evidence.get("mean_state")
            precision_state = evidence.get("precision_state")
            if not isinstance(mean_state, dict) or not isinstance(precision_state, dict):
                continue
            if not any(key in mean_state and key in precision_state for key in expert_keys):
                continue
            payloads.append({
                "num_batches": int(evidence.get("num_batches", 0)),
                "mean_state": mean_state,
                "precision_state": precision_state,
            })
        return payloads

    def _get_or_init_prior_state(self, updated_bayes_state, layer_id, expert_id, expert_keys, global_state):
        expert_state = get_bayes_expert_state(updated_bayes_state, layer_id, expert_id)
        if expert_state is not None:
            return expert_state
        layer_state = updated_bayes_state.setdefault("experts", {}).setdefault(str(layer_id), {})
        log_gamma0 = math.log(self.gamma0_init)
        log_precision_state = {}
        for key in expert_keys:
            value = global_state[key].detach().cpu()
            log_precision_state[key] = (
                torch.full_like(value, fill_value=log_gamma0)
                if torch.is_floating_point(value)
                else value.clone()
            )
        layer_state[str(expert_id)] = {
            "log_precision_state": log_precision_state,
            "log_n0": torch.tensor(math.log(self.n0_init), dtype=torch.float32),
        }
        return layer_state[str(expert_id)]

    def _get_prior_n0(self, prior_state):
        log_n0 = prior_state.get("log_n0")
        if log_n0 is None:
            return torch.tensor(self.n0_init, dtype=torch.float32)
        return torch.exp(log_n0.detach().cpu().float()).clamp(min=self.min_precision)


class DecoupledMoEAggregator(Aggregator):
    def __init__(self, args):
        self.args = args
        self.non_expert_agg_method = str(getattr(args, "non_expert_agg_method", "fedavg")).lower()
        self.expert_agg_method = str(getattr(args, "expert_agg_method", "expert_fedavg")).lower()
        self._validate_methods()
        self.non_expert_aggregator = self._build_full_aggregator(
            self.non_expert_agg_method,
            for_expert=False,
        )
        self.expert_aggregator = self._build_full_aggregator(
            self.expert_agg_method,
            for_expert=True,
        )
        self._logged_methods = False

    def _validate_methods(self):
        non_expert_methods = {"client_avg", "fedavg", "expert_fedavg"}
        expert_methods = {"client_avg", "fedavg", "expert_fedavg", "expert_bayes_meta"}
        if self.non_expert_agg_method not in non_expert_methods:
            raise ValueError(
                "Unsupported non_expert_agg_method: "
                f"{self.non_expert_agg_method}. Supported options: "
                "client_avg, fedavg, expert_fedavg"
            )
        if self.expert_agg_method not in expert_methods:
            raise ValueError(
                f"Unsupported expert_agg_method: {self.expert_agg_method}. Supported options: "
                "client_avg, fedavg, expert_fedavg, expert_bayes_meta"
            )

    def _build_full_aggregator(self, method, for_expert):
        if method == "client_avg":
            return ClientAvgAggregator()
        if method == "fedavg":
            return FedAvgAggregator()
        if method == "expert_fedavg":
            return ExpertFedAvgAggregator() if for_expert else FedAvgAggregator()
        if method == "expert_bayes_meta":
            if not for_expert:
                raise ValueError("expert_bayes_meta is only supported for expert parameters")
            return ExpertBayesMetaAggregator(self.args)
        raise ValueError(f"Unsupported aggregation method: {method}")

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        if len(client_updates) == 0:
            raise ValueError("DecoupledMoE requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        self._log_methods_once(kwargs.get("logger"))
        non_expert_state = self.non_expert_aggregator.aggregate(
            client_updates=client_updates,
            client_weights=client_weights,
            global_model=global_model,
            **kwargs,
        )
        expert_output = self.expert_aggregator.aggregate(
            client_updates=client_updates,
            client_weights=client_weights,
            global_model=global_model,
            **kwargs,
        )
        expert_state = (
            expert_output["model_state"]
            if self.expert_agg_method == "expert_bayes_meta"
            else expert_output
        )

        aggregated_state = collections.OrderedDict()
        for key in client_updates[0].keys():
            is_expert_key = parse_expert_ref(key) is not None
            aggregated_state[key] = expert_state[key] if is_expert_key else non_expert_state[key]

        if self.expert_agg_method == "expert_bayes_meta":
            aggregation_output = dict(expert_output)
            aggregation_output["model_state"] = aggregated_state
            return aggregation_output
        return aggregated_state

    def _log_methods_once(self, logger):
        if self._logged_methods:
            return
        if logger is None:
            try:
                logger = logging.getLogger(get_experiment_stem(self.args))
            except AttributeError:
                logger = logging.getLogger(__name__)
        logger.info(f"--decoupled_moe_non_expert_agg_method : {self.non_expert_agg_method}")
        logger.info(f"--decoupled_moe_expert_agg_method : {self.expert_agg_method}")
        self._logged_methods = True


def build_aggregator(args):
    if args.agg_method == "decoupled_moe":
        return DecoupledMoEAggregator(args)
    if args.agg_method == "expert_bayes_meta":
        return ExpertBayesMetaAggregator(args)
    if args.agg_method == "expert_fedavg":
        return ExpertFedAvgAggregator()
    if args.agg_method == "client_avg":
        return ClientAvgAggregator()
    if args.agg_method == "fedavg":
        return FedAvgAggregator()
    raise ValueError(f"Unknown aggregation method: {args.agg_method}")
