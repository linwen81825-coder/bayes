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
    # 服务器主导的 expert Bayesian aggregation：
    # - 非 expert / router 参数继续做标准 FedAvg；
    # - expert 参数先根据 server prior 与 client evidence 计算最优局部后验；
    # - 再用这些局部后验近似替代完整 ELBO 中的局部解，更新全局 expert。
    def __init__(self, args):
        self.args = args
        self.base_fedavg = FedAvgAggregator()
        self.min_precision = 1e-6
        self.max_precision = max(float(getattr(args, "bayes_ai_max", 1e3)), self.min_precision)
        self.max_n0 = 1e6
        self.gamma0_init = max(float(getattr(args, "bayes_gamma0_init", 1.0)), self.min_precision)
        self.n0_init = max(float(getattr(args, "bayes_n0_init", 1.0)), self.min_precision)
        self.meta_steps = max(int(getattr(args, "bayes_meta_steps", 5)), 1)
        self.meta_lr = float(getattr(args, "bayes_meta_lr", 0.001))
        self.update_precision = bool(getattr(args, "bayes_update_precision", True))
        self.update_strength = bool(getattr(args, "bayes_update_strength", True))
        self.meta_device = self._resolve_meta_device(args)
        self.empty_cache_after_aggregation = bool(
            getattr(args, "bayes_empty_cache_after_aggregation", False)
        )
        self.client_weight_mode = str(
            getattr(args, "bayes_client_weight_mode", "uniform")
        ).lower()
        if self.client_weight_mode not in {"uniform", "sqrt_usage", "usage"}:
            raise ValueError(
                "bayes_client_weight_mode must be one of: "
                "uniform, sqrt_usage, usage"
            )
        self.direction_diag = bool(getattr(args, "bayes_direction_diag", False))
        self.direction_diag_detail = bool(getattr(args, "bayes_direction_diag_detail", False))
        print(
            "[ExpertBayesMetaAggregator] "
            f"bayes_meta_device={self.meta_device} "
            f"args.device={getattr(args, 'device', 'cpu')} "
            f"cuda_available={torch.cuda.is_available()}"
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
        expert_weights = kwargs.get("expert_weights")

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
        old_expert_state = (
            self._clone_expert_state_for_direction_diag(global_state, expert_groups)
            if self.direction_diag
            else None
        )
        metrics = {
            "updated_experts": 0,
            "skipped_experts": 0,
            "evidence_clients": 0,
            "expert_param_groups": len(expert_groups),
            "local_posteriors": 0,
            "bayes_meta_device": str(self.meta_device),
            "bayes_cuda_available": bool(torch.cuda.is_available()),
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

        if self.direction_diag:
            direction_summary = self._log_direction_diagnostics(
                old_expert_state=old_expert_state,
                aggregated_state=aggregated_state,
                client_updates=client_updates,
                client_weights=client_weights,
                client_expert_usages=expert_weights,
                expert_groups=expert_groups,
            )
            metrics["bayes_vs_fedavg_direction_summary"] = direction_summary

        updated_bayes_state["round"] = int(updated_bayes_state.get("round", 0)) + 1
        metrics["bayes_aggregation_time_sec"] = round(
            time.perf_counter() - aggregate_start_time,
            4,
        )
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
        client_payloads = self._collect_client_payloads(
            expert_evidence=expert_evidence,
            layer_id=layer_id,
            expert_id=expert_id,
        )
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
            expert_metric["expert_meta_time_sec"] = round(
                time.perf_counter() - expert_start_time,
                4,
            )
            return {
                key: global_state[key].detach().cpu().clone()
                for key in expert_keys
            }, 0, 0, expert_metric

        total_usage = sum(payload["usage"] for payload in client_payloads)
        prior_n0 = self._get_prior_n0(prior_state)
        optimized_mean_state, optimized_log_precision_state, optimized_log_n0, local_posterior_count, meta_loss = (
            self._optimize_expert_prior(
                expert_keys=expert_keys,
                global_state=global_state,
                prior_state=prior_state,
                prior_n0=prior_n0,
                client_payloads=client_payloads,
                layer_id=layer_id,
                expert_id=expert_id,
            )
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
        expert_metric["expert_meta_time_sec"] = round(
            time.perf_counter() - expert_start_time,
            4,
        )
        return aggregated_params, len(client_payloads), local_posterior_count, expert_metric

    def _optimize_expert_prior(
        self,
        expert_keys,
        global_state,
        prior_state,
        prior_n0,
        client_payloads,
        layer_id=None,
        expert_id=None,
    ):
        prior_mean_params = collections.OrderedDict()
        log_precision_params = collections.OrderedDict()
        optim_params = []

        for key in expert_keys:
            mean_param = torch.nn.Parameter(
                global_state[key]
                .detach()
                .to(device=self.meta_device, dtype=torch.float32)
                .clone()
            )
            prior_mean_params[key] = mean_param
            optim_params.append(mean_param)

            reference_tensor = (
                global_state[key]
                .detach()
                .to(device=self.meta_device, dtype=torch.float32)
                .clone()
            )
            log_precision = prior_state.get("log_precision_state", {}).get(key)
            if log_precision is None or not torch.is_floating_point(log_precision):
                log_precision = torch.full_like(
                    reference_tensor,
                    fill_value=math.log(self.gamma0_init),
                    device=self.meta_device,
                )
            else:
                log_precision = (
                    log_precision
                    .detach()
                    .to(device=self.meta_device, dtype=torch.float32)
                    .clone()
                )

            log_precision_param = torch.nn.Parameter(
                log_precision,
                requires_grad=self.update_precision,
            )
            log_precision_params[key] = log_precision_param
            if self.update_precision:
                optim_params.append(log_precision_param)

        log_n0_value = prior_state.get("log_n0")
        if log_n0_value is None:
            log_n0_value = torch.tensor(
                math.log(max(float(prior_n0.item()), self.min_precision)),
                dtype=torch.float32,
                device=self.meta_device,
            )
        else:
            log_n0_value = (
                log_n0_value
                .detach()
                .to(device=self.meta_device, dtype=torch.float32)
                .clone()
            )
        log_n0_param = torch.nn.Parameter(
            log_n0_value,
            requires_grad=self.update_strength,
        )
        if self.update_strength:
            optim_params.append(log_n0_param)

        optimizer = torch.optim.Adam(optim_params, lr=self.meta_lr)
        local_posterior_count = 0
        last_finite_meta_loss_value = None
        last_finite_local_posterior_count = 0
        encountered_nonfinite = False
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
                encountered_nonfinite = True
                layer_label = "unknown" if layer_id is None else str(layer_id)
                expert_label = "unknown" if expert_id is None else str(expert_id)
                print(
                    "[ExpertBayesMetaAggregator] warning: non-finite meta_loss "
                    f"layer={layer_label} expert={expert_label}; "
                    "skipping backward and optimizer.step, keeping last finite prior parameters."
                )
                break
            meta_loss.backward()
            optimizer.step()
            self._project_meta_params(log_precision_params, log_n0_param)

            last_finite_meta_loss_value = float(meta_loss.detach().cpu().item())
            last_finite_local_posterior_count = local_posterior_count

        final_local_posterior_count = local_posterior_count
        final_meta_loss_value = float("nan")
        if not encountered_nonfinite:
            with torch.no_grad():
                final_meta_loss, final_local_posterior_count = self._compute_expert_meta_loss(
                    expert_keys=expert_keys,
                    prior_mean_params=prior_mean_params,
                    log_precision_params=log_precision_params,
                    log_n0_param=log_n0_param,
                    client_payloads=client_payloads,
                )
            if torch.isfinite(final_meta_loss).item():
                final_meta_loss_value = float(final_meta_loss.detach().cpu().item())
            elif last_finite_meta_loss_value is not None:
                final_meta_loss_value = last_finite_meta_loss_value
                final_local_posterior_count = last_finite_local_posterior_count
            else:
                final_meta_loss_value = float("nan")
        else:
            if last_finite_meta_loss_value is not None:
                final_meta_loss_value = last_finite_meta_loss_value
                final_local_posterior_count = last_finite_local_posterior_count

        return (
            prior_mean_params,
            log_precision_params,
            log_n0_param,
            final_local_posterior_count,
            final_meta_loss_value,
        )

    def _summarize_client_payloads(self, client_payloads, expert_keys):
        if not client_payloads:
            return {
                "bayes_client_weight_mode": self.client_weight_mode,
                "usage_total": 0.0,
                "usage_max": 0.0,
                "usage_weight_sum": 0.0,
                "usage_weight_min": 0.0,
                "usage_weight_max": 0.0,
                "usage_weight_mean": 0.0,
                "usage_weight_max_ratio": 0.0,
                "num_batches_total": 0,
                "num_batches_mean": 0.0,
                "local_precision_mean": None,
                "local_precision_min": None,
                "local_precision_max": None,
                "local_precision_low_pct": None,
                "local_precision_high_pct": None,
            }

        usage_values = [float(payload.get("usage", 0.0)) for payload in client_payloads]
        usage_weight_values = [
            self._get_client_meta_weight(payload)
            for payload in client_payloads
        ]
        usage_weight_sum = float(sum(usage_weight_values))
        batch_values = [int(payload.get("num_batches", 0)) for payload in client_payloads]
        precision_values = []
        for payload in client_payloads:
            precision_state = payload.get("precision_state", {})
            for key in expert_keys:
                value = precision_state.get(key)
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    precision_values.append(value.detach().cpu().float().reshape(-1))

        summary = {
            "bayes_client_weight_mode": self.client_weight_mode,
            "usage_total": round(float(sum(usage_values)), 6),
            "usage_max": round(float(max(usage_values)), 6),
            "usage_weight_sum": round(usage_weight_sum, 6),
            "usage_weight_min": round(float(min(usage_weight_values)), 6),
            "usage_weight_max": round(float(max(usage_weight_values)), 6),
            "usage_weight_mean": round(
                usage_weight_sum / max(len(usage_weight_values), 1),
                6,
            ),
            "usage_weight_max_ratio": round(
                float(max(usage_weight_values) / max(usage_weight_sum, 1e-12)),
                6,
            ),
            "num_batches_total": int(sum(batch_values)),
            "num_batches_mean": round(float(sum(batch_values) / max(len(batch_values), 1)), 6),
        }
        if not precision_values:
            summary.update(
                {
                    "local_precision_mean": None,
                    "local_precision_min": None,
                    "local_precision_max": None,
                    "local_precision_low_pct": None,
                    "local_precision_high_pct": None,
                }
            )
            return summary

        precision_vector = torch.cat(precision_values)
        low_threshold = 1.0001e-4
        high_threshold = 0.9999 * self.max_precision
        summary.update(
            {
                "local_precision_mean": round(float(precision_vector.mean().item()), 6),
                "local_precision_min": round(float(precision_vector.min().item()), 6),
                "local_precision_max": round(float(precision_vector.max().item()), 6),
                "local_precision_low_pct": round(
                    float((precision_vector <= low_threshold).float().mean().item()),
                    6,
                ),
                "local_precision_high_pct": round(
                    float((precision_vector >= high_threshold).float().mean().item()),
                    6,
                ),
            }
        )
        return summary

    def _summarize_mean_update(self, global_state, optimized_mean_state, expert_keys):
        if global_state is None or optimized_mean_state is None:
            return {
                "param_delta_norm": None,
                "param_prior_norm": None,
                "param_delta_rel": None,
            }

        delta_sq = 0.0
        prior_sq = 0.0
        for key in expert_keys:
            prior_value = global_state[key].detach().cpu().float()
            updated_value = optimized_mean_state[key].detach().cpu().float()
            delta_sq += float((updated_value - prior_value).square().sum().item())
            prior_sq += float(prior_value.square().sum().item())

        delta_norm = math.sqrt(max(delta_sq, 0.0))
        prior_norm = math.sqrt(max(prior_sq, 0.0))
        delta_rel = delta_norm / max(prior_norm, 1e-12)
        return {
            "param_delta_norm": round(delta_norm, 6),
            "param_prior_norm": round(prior_norm, 6),
            "param_delta_rel": round(delta_rel, 6),
        }

    def _get_client_meta_weight(self, payload):
        usage = max(float(payload.get("usage", 0.0)), 1.0)
        if self.client_weight_mode == "uniform":
            return 1.0
        if self.client_weight_mode == "sqrt_usage":
            return math.sqrt(usage)
        if self.client_weight_mode == "usage":
            return usage
        raise ValueError(f"Unknown bayes_client_weight_mode: {self.client_weight_mode}")

    def _compute_expert_meta_loss(
        self,
        expert_keys,
        prior_mean_params,
        log_precision_params,
        log_n0_param,
        client_payloads,
    ):
        zero = next(iter(prior_mean_params.values())).new_tensor(0.0)
        weighted_losses = []
        weight_values = []
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
                target_device = prior_mean.device
                target_dtype = prior_mean.dtype
                local_mean = local_mean.detach().to(
                    device=target_device,
                    dtype=target_dtype,
                )
                local_precision = local_precision.detach().to(
                    device=target_device,
                    dtype=target_dtype,
                )
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

            # uniform 对应算法文档里的 |S_k|^{-1} sum_i；
            # sqrt_usage / usage 只替换 client evidence 的聚合权重，
            # 不改变每个客户端局部 posterior 与二次项的计算公式。
            if has_local_terms:
                weight = self._get_client_meta_weight(payload)
                weight_tensor = client_loss.new_tensor(weight)
                weighted_losses.append(client_loss * weight_tensor)
                weight_values.append(weight_tensor)

        if len(weighted_losses) == 0:
            return zero, 0

        total_weight = torch.stack(weight_values).sum().clamp_min(1e-12)
        meta_loss = torch.stack(weighted_losses).sum() / total_weight
        return meta_loss, local_posterior_count

    def _project_meta_params(self, log_precision_params, log_n0_param):
        log_min_precision = math.log(self.min_precision)
        log_max_precision = math.log(self.max_precision)
        log_max_n0 = math.log(self.max_n0)
        with torch.no_grad():
            for log_precision_param in log_precision_params.values():
                log_precision_param.clamp_(min=log_min_precision, max=log_max_precision)
            log_n0_param.clamp_(min=log_min_precision, max=log_max_n0)

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
        if optimized_log_precision_state is None:
            log_precision_state = prior_state.get("log_precision_state", {})
        else:
            log_precision_state = optimized_log_precision_state

        gamma_values = []
        for value in log_precision_state.values():
            if not torch.is_tensor(value) or not torch.is_floating_point(value):
                continue
            gamma_values.append(torch.exp(value.detach().cpu().float()).reshape(-1))

        if gamma_values:
            gamma_vector = torch.cat(gamma_values)
            avg_gamma0 = float(gamma_vector.mean().item())
            min_gamma0 = float(gamma_vector.min().item())
            max_gamma0 = float(gamma_vector.max().item())
        else:
            avg_gamma0 = float(self.gamma0_init)
            min_gamma0 = float(self.gamma0_init)
            max_gamma0 = float(self.gamma0_init)

        if optimized_log_n0 is None:
            log_n0 = prior_state.get("log_n0")
            if log_n0 is None:
                n0 = float(self.n0_init)
            else:
                n0 = float(torch.exp(log_n0.detach().cpu().float()).item())
        else:
            n0 = float(torch.exp(optimized_log_n0.detach().cpu().float()).item())

        expert_keys = expert_keys or []
        client_payloads = client_payloads or []
        payload_summary = self._summarize_client_payloads(client_payloads, expert_keys)
        update_summary = self._summarize_mean_update(global_state, optimized_mean_state, expert_keys)

        metric = {
            "status": status,
            "layer_id": str(layer_id),
            "expert_id": str(expert_id),
            "bayes_meta_device": str(self.meta_device),
            "clients": int(contributing_clients),
            "local_posteriors": int(local_posterior_count),
            "meta_loss": None if meta_loss is None else round(float(meta_loss), 6),
            "n0": round(n0, 6),
            "avg_gamma0": round(avg_gamma0, 6),
            "min_gamma0": round(min_gamma0, 6),
            "max_gamma0": round(max_gamma0, 6),
        }
        if optimized_mean_state:
            metric["optimized_mean_device"] = str(
                next(iter(optimized_mean_state.values())).device
            )
        if optimized_log_precision_state:
            metric["log_precision_device"] = str(
                next(iter(optimized_log_precision_state.values())).device
            )
        if optimized_log_n0 is not None:
            metric["log_n0_device"] = str(optimized_log_n0.device)
        metric.update(payload_summary)
        metric.update(update_summary)
        return metric

    def _collect_client_payloads(self, expert_evidence, layer_id, expert_id):
        payloads = []
        for client_evidence in expert_evidence:
            evidence = get_client_expert_evidence(
                client_evidence=client_evidence,
                layer_id=layer_id,
                expert_id=expert_id,
            )
            if evidence is None:
                continue

            usage = float(evidence.get("usage", 0.0))
            if usage <= 0:
                continue

            payloads.append(
                {
                    "usage": usage,
                    "num_batches": int(evidence.get("num_batches", 0)),
                    "mean_state": evidence.get("mean_state", {}),
                    "precision_state": evidence.get("precision_state", {}),
                }
            )
        return payloads

    def _clone_expert_state_for_direction_diag(self, global_state, expert_groups):
        old_state = {}
        for expert_keys in expert_groups.values():
            for key in expert_keys:
                value = global_state.get(key)
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    old_state[key] = value.detach().cpu().clone()
        return old_state

    def _get_direction_diag_usage(self, client_usage, layer_id, expert_id):
        if client_usage is None:
            return None
        expert_index = int(expert_id)
        if isinstance(client_usage, dict):
            layer_stats = client_usage.get("expert_stats_by_layer", {}).get(str(layer_id), {})
            usage = layer_stats.get("expert_activations")
            if usage is None:
                usage = client_usage.get("expert_activations_by_layer", {}).get(str(layer_id))
            if usage is None:
                usage = client_usage.get("expert_activations") if layer_id is None else None
            if usage is None or expert_index >= len(usage):
                return None
            return max(float(usage[expert_index]), 0.0)

        if expert_index >= len(client_usage):
            return None
        return max(float(client_usage[expert_index]), 0.0)

    def _get_direction_diag_fedavg_weights(
        self,
        layer_id,
        expert_id,
        client_weights,
        client_expert_usages,
        num_clients,
    ):
        usage_weights = []
        missing_usage = False
        for client_idx in range(num_clients):
            client_usage = None
            if isinstance(client_expert_usages, list) and client_idx < len(client_expert_usages):
                client_usage = client_expert_usages[client_idx]
            usage = self._get_direction_diag_usage(client_usage, layer_id, expert_id)
            if usage is None:
                missing_usage = True
                usage = 0.0
            usage_weights.append(float(usage))

        fallback_mode = None
        weights = usage_weights
        total_weight = float(sum(weights))
        if missing_usage or total_weight <= 0.0:
            sample_weights = [float(weight) for weight in client_weights]
            sample_total = float(sum(sample_weights))
            if sample_total > 0.0:
                weights = sample_weights
                total_weight = sample_total
                fallback_mode = "sample_count"
            else:
                weights = [1.0 for _ in range(num_clients)]
                total_weight = float(num_clients)
                fallback_mode = "uniform"

        if total_weight <= 0.0:
            weights = [1.0 for _ in range(num_clients)]
            total_weight = float(num_clients)
            fallback_mode = "uniform"

        max_ratio = max(weights) / max(total_weight, 1e-12) if weights else 0.0
        return weights, total_weight, max_ratio, fallback_mode

    def _compute_direction_metrics_for_expert(
        self,
        layer_id,
        expert_id,
        expert_keys,
        old_expert_state,
        aggregated_state,
        client_updates,
        client_weights,
        client_expert_usages,
    ):
        """Compare Bayes expert update direction with a diagnostic ExpertFedAvg candidate.

        cos_bayes_fedavg close to 1 means Bayes and FedAvg move in the same direction.
        A value close to 0 means weakly related directions; a negative value means conflict.
        A small bayes_vs_fedavg_rel means Bayes and FedAvg are practically similar.
        A large bayes_vs_fedavg_rel with poor test accuracy can indicate over-deviation.
        bayes_delta_rel > fedavg_delta_rel means Bayes is more aggressive; smaller means conservative.
        """
        weights, weight_sum, weight_max_ratio, fallback_mode = self._get_direction_diag_fedavg_weights(
            layer_id=layer_id,
            expert_id=expert_id,
            client_weights=client_weights,
            client_expert_usages=client_expert_usages,
            num_clients=len(client_updates),
        )
        normalized_weights = [weight / max(weight_sum, 1e-12) for weight in weights]

        old_norm2 = 0.0
        fedavg_norm2 = 0.0
        fedavg_delta_norm2 = 0.0
        bayes_delta_norm2 = 0.0
        bayes_vs_fedavg_norm2 = 0.0
        dot = 0.0
        skipped_key_count = 0
        valid_key_count = 0

        with torch.no_grad():
            for key in expert_keys:
                old_value = old_expert_state.get(key)
                bayes_value = aggregated_state.get(key)
                if old_value is None or bayes_value is None:
                    skipped_key_count += 1
                    continue
                if not torch.is_tensor(old_value) or not torch.is_tensor(bayes_value):
                    skipped_key_count += 1
                    continue
                if not torch.is_floating_point(old_value) or not torch.is_floating_point(bayes_value):
                    skipped_key_count += 1
                    continue
                if any(key not in update for update in client_updates):
                    skipped_key_count += 1
                    continue

                old_tensor = old_value.detach().cpu().float()
                bayes_tensor = bayes_value.detach().cpu().float()
                fedavg_tensor = torch.zeros_like(old_tensor)
                for update, weight in zip(client_updates, normalized_weights):
                    update_value = update[key]
                    if not torch.is_tensor(update_value) or not torch.is_floating_point(update_value):
                        continue
                    fedavg_tensor.add_(update_value.detach().cpu().float(), alpha=float(weight))

                fedavg_delta = fedavg_tensor - old_tensor
                bayes_delta = bayes_tensor - old_tensor
                bayes_vs_fedavg = bayes_tensor - fedavg_tensor

                old_norm2 += float(old_tensor.square().sum().item())
                fedavg_norm2 += float(fedavg_tensor.square().sum().item())
                fedavg_delta_norm2 += float(fedavg_delta.square().sum().item())
                bayes_delta_norm2 += float(bayes_delta.square().sum().item())
                bayes_vs_fedavg_norm2 += float(bayes_vs_fedavg.square().sum().item())
                dot += float((bayes_delta * fedavg_delta).sum().item())
                valid_key_count += 1

        if valid_key_count == 0:
            return {
                "layer": str(layer_id),
                "expert": str(expert_id),
                "skipped": True,
                "skipped_key_count": int(skipped_key_count),
                "fedavg_weight_sum": round(float(weight_sum), 6),
                "fedavg_weight_max_ratio": round(float(weight_max_ratio), 6),
                "fedavg_weight_fallback": fallback_mode is not None,
                "fedavg_weight_fallback_mode": fallback_mode,
            }

        eps = 1e-12
        old_norm = math.sqrt(max(old_norm2, 0.0))
        fedavg_norm = math.sqrt(max(fedavg_norm2, 0.0))
        fedavg_delta_norm = math.sqrt(max(fedavg_delta_norm2, 0.0))
        bayes_delta_norm = math.sqrt(max(bayes_delta_norm2, 0.0))
        bayes_vs_fedavg_norm = math.sqrt(max(bayes_vs_fedavg_norm2, 0.0))
        cos_valid = fedavg_delta_norm > eps and bayes_delta_norm > eps
        cos_value = None
        if cos_valid:
            cos_value = dot / max(bayes_delta_norm * fedavg_delta_norm, eps)
            cos_value = max(min(float(cos_value), 1.0), -1.0)

        return {
            "layer": str(layer_id),
            "expert": str(expert_id),
            "skipped": False,
            "skipped_key_count": int(skipped_key_count),
            "valid_key_count": int(valid_key_count),
            "old_norm": round(float(old_norm), 6),
            "fedavg_delta_rel": round(float(fedavg_delta_norm / max(old_norm, eps)), 6),
            "bayes_delta_rel": round(float(bayes_delta_norm / max(old_norm, eps)), 6),
            "bayes_vs_fedavg_rel": round(float(bayes_vs_fedavg_norm / max(fedavg_norm, eps)), 6),
            "cos_bayes_fedavg": None if cos_value is None else round(float(cos_value), 6),
            "cos_valid": bool(cos_valid),
            "fedavg_weight_sum": round(float(weight_sum), 6),
            "fedavg_weight_max_ratio": round(float(weight_max_ratio), 6),
            "fedavg_weight_fallback": fallback_mode is not None,
            "fedavg_weight_fallback_mode": fallback_mode,
        }

    def _summary_stat(self, values):
        finite_values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        if not finite_values:
            return None, None, None
        return (
            round(sum(finite_values) / len(finite_values), 6),
            round(min(finite_values), 6),
            round(max(finite_values), 6),
        )

    def _summarize_direction_metrics(self, details):
        valid_details = [detail for detail in details if not detail.get("skipped")]
        cos_values = [
            detail.get("cos_bayes_fedavg")
            for detail in valid_details
            if detail.get("cos_valid") and detail.get("cos_bayes_fedavg") is not None
        ]
        fedavg_delta_mean, fedavg_delta_min, fedavg_delta_max = self._summary_stat(
            [detail.get("fedavg_delta_rel") for detail in valid_details]
        )
        bayes_delta_mean, bayes_delta_min, bayes_delta_max = self._summary_stat(
            [detail.get("bayes_delta_rel") for detail in valid_details]
        )
        bayes_vs_mean, bayes_vs_min, bayes_vs_max = self._summary_stat(
            [detail.get("bayes_vs_fedavg_rel") for detail in valid_details]
        )
        cos_mean, cos_min, cos_max = self._summary_stat(cos_values)
        weight_ratio_mean, _, weight_ratio_max = self._summary_stat(
            [detail.get("fedavg_weight_max_ratio") for detail in details]
        )

        return {
            "enabled": True,
            "num_experts": int(len(details)),
            "num_valid_cos": int(len(cos_values)),
            "skipped_expert_count": int(sum(1 for detail in details if detail.get("skipped"))),
            "skipped_key_count": int(sum(int(detail.get("skipped_key_count", 0)) for detail in details)),
            "fedavg_delta_rel_mean": fedavg_delta_mean,
            "fedavg_delta_rel_min": fedavg_delta_min,
            "fedavg_delta_rel_max": fedavg_delta_max,
            "bayes_delta_rel_mean": bayes_delta_mean,
            "bayes_delta_rel_min": bayes_delta_min,
            "bayes_delta_rel_max": bayes_delta_max,
            "bayes_vs_fedavg_rel_mean": bayes_vs_mean,
            "bayes_vs_fedavg_rel_min": bayes_vs_min,
            "bayes_vs_fedavg_rel_max": bayes_vs_max,
            "cos_bayes_fedavg_mean": cos_mean,
            "cos_bayes_fedavg_min": cos_min,
            "cos_bayes_fedavg_max": cos_max,
            "cos_negative_count": int(sum(1 for value in cos_values if value < 0.0)),
            "cos_low_count": int(sum(1 for value in cos_values if value < 0.2)),
            "cos_mid_count": int(sum(1 for value in cos_values if 0.2 <= value <= 0.8)),
            "cos_high_count": int(sum(1 for value in cos_values if value > 0.8)),
            "fedavg_weight_fallback_count": int(
                sum(1 for detail in details if detail.get("fedavg_weight_fallback"))
            ),
            "fedavg_weight_max_ratio_mean": weight_ratio_mean,
            "fedavg_weight_max_ratio_max": weight_ratio_max,
        }

    def _emit_direction_diag_log(self, prefix, payload):
        message = f"{prefix} : {payload}"
        emitted = False
        for logger_obj in logging.Logger.manager.loggerDict.values():
            if not isinstance(logger_obj, logging.Logger):
                continue
            if not any(isinstance(handler, logging.FileHandler) for handler in logger_obj.handlers):
                continue
            logger_obj.info(message)
            emitted = True
        if not emitted:
            logging.getLogger(__name__).info(message)

    def _log_direction_diagnostics(
        self,
        old_expert_state,
        aggregated_state,
        client_updates,
        client_weights,
        client_expert_usages,
        expert_groups,
    ):
        details = []
        for (layer_id, expert_id), expert_keys in expert_groups.items():
            detail = self._compute_direction_metrics_for_expert(
                layer_id=layer_id,
                expert_id=expert_id,
                expert_keys=expert_keys,
                old_expert_state=old_expert_state or {},
                aggregated_state=aggregated_state,
                client_updates=client_updates,
                client_weights=client_weights,
                client_expert_usages=client_expert_usages,
            )
            details.append(detail)

        summary = self._summarize_direction_metrics(details)
        self._emit_direction_diag_log("--bayes_vs_fedavg_direction_summary", summary)

        if self.direction_diag_detail:
            for detail in details[:32]:
                self._emit_direction_diag_log("--bayes_vs_fedavg_direction_detail", detail)

        return summary

    def _get_or_init_prior_state(self, updated_bayes_state, layer_id, expert_id, expert_keys, global_state):
        expert_state = get_bayes_expert_state(
            bayes_state=updated_bayes_state,
            layer_id=layer_id,
            expert_id=expert_id,
        )
        if expert_state is not None:
            return expert_state

        experts_by_layer = updated_bayes_state.setdefault("experts", {})
        layer_state = experts_by_layer.setdefault(str(layer_id), {})
        log_precision_state = {}
        log_gamma0 = math.log(self.gamma0_init)
        for key in expert_keys:
            value = global_state[key].detach().cpu()
            if torch.is_floating_point(value):
                log_precision_state[key] = torch.full_like(value, fill_value=log_gamma0)
            else:
                log_precision_state[key] = value.clone()

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

    def _get_prior_precision(self, prior_state, key, reference_tensor):
        log_precision_state = prior_state.get("log_precision_state", {})
        prior_log_precision = log_precision_state.get(key)
        if prior_log_precision is None or not torch.is_floating_point(prior_log_precision):
            return torch.full_like(reference_tensor, fill_value=self.gamma0_init)
        return torch.exp(prior_log_precision.detach().cpu().to(dtype=reference_tensor.dtype)).clamp(
            min=self.min_precision
        )


def build_aggregator(args):
    if args.agg_method == "expert_bayes_meta":
        return ExpertBayesMetaAggregator(args)
    if args.agg_method == "expert_fedavg":
        return ExpertFedAvgAggregator()
    if args.agg_method == "fedavg":
        return FedAvgAggregator()
    raise ValueError(f"Unknown aggregation method: {args.agg_method}")
