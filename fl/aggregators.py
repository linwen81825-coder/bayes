import collections
import math
from abc import ABC, abstractmethod

import torch
from torch import nn

from fl.pism import (
    ExpertPISM,
    build_grad_dot_pism_feature_tensor,
    build_pism_feature_tensor,
    normalize_pism_inputs,
)
from fl.uoc_foga import (
    average_delta_states,
    build_client_grad_query_for_expert,
    build_stratified_query_for_expert,
    build_reference_query_for_layer,
    cosine_delta_to_delta,
    cosine_grad_to_grad,
    delta_to_negative_grad_score,
    dot_grad_to_grad,
    expert_weight_entropy,
    extract_expert_delta_state,
    l2_norm_state,
    positive_score_to_weights,
    summarize_scores,
)


class Aggregator(ABC):
    # 聚合器统一接口。后续新增聚合方法时，优先在本文件扩展参数级聚合逻辑。
    @abstractmethod
    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        pass

    def get_checkpoint_state(self):
        # 默认聚合器没有跨 round 状态，checkpoint 中保存空状态即可。
        return {}

    def load_checkpoint_state(self, state, map_location=None):
        # 旧聚合器没有可恢复状态；保留 hook 让 server 统一调用。
        return


def build_client_weights(method, client_sample_counts):
    if method == "sample_weighted":
        weights = [float(value) for value in client_sample_counts]
    elif method == "uniform":
        weights = [1.0 for _ in client_sample_counts]
    else:
        raise ValueError(f"Unknown aggregation method: {method!r}")

    total = sum(weights)
    if total <= 0:
        raise ValueError(f"Aggregation method {method!r} requires positive total weight")

    return [weight / total for weight in weights]


def is_expert_parameter(key: str) -> bool:
    parts = key.split(".")
    return "experts" in parts


def parse_expert_parameter_key(key: str):
    parts = key.split(".")
    if "blocks" not in parts or "experts" not in parts:
        return None

    block_index = parts.index("blocks")
    expert_index = parts.index("experts")
    if block_index + 1 >= len(parts) or expert_index + 1 >= len(parts):
        return None

    layer_id = parts[block_index + 1]
    expert_id = parts[expert_index + 1]
    if not layer_id.isdigit() or not expert_id.isdigit():
        return None
    return layer_id, expert_id


class SplitParameterAggregator(Aggregator):
    # 按参数名拆分聚合链路：专家参数和非专家参数可以使用不同的客户端权重策略。
    def __init__(self, non_expert_method: str, expert_method: str):
        self.non_expert_method = non_expert_method
        self.expert_method = expert_method
        self.last_aggregation_metrics = {}

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        if len(client_updates) == 0:
            raise ValueError("SplitParameterAggregator requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        non_expert_weights = build_client_weights(self.non_expert_method, client_weights)
        expert_weights = build_client_weights(self.expert_method, client_weights)
        expert_client_weights = {
            str(client_idx): float(weight)
            for client_idx, weight in enumerate(expert_weights)
        }
        expert_weights_by_layer = {}

        aggregated_state = collections.OrderedDict()
        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()
            if not torch.is_floating_point(first_value):
                # 非浮点 buffer 通常不能加权平均，沿用第一个客户端的值。
                aggregated_state[key] = first_value.clone()
                continue

            parsed_expert_key = parse_expert_parameter_key(key)
            if parsed_expert_key is not None:
                layer_id, expert_id = parsed_expert_key
                expert_weights_by_layer.setdefault(str(layer_id), {})[str(expert_id)] = dict(
                    expert_client_weights
                )

            if is_expert_parameter(key):
                normalized_weights = expert_weights
            else:
                normalized_weights = non_expert_weights

            aggregated_state[key] = torch.zeros_like(first_value)
            for update, weight in zip(client_updates, normalized_weights):
                aggregated_state[key] += update[key].detach().cpu() * weight

        self.last_aggregation_metrics = {
            "expert_aggregation_weights": {
                "weight_source": self.expert_method,
                "shared_across_experts": True,
                "client_weights": expert_client_weights,
                "by_layer": expert_weights_by_layer,
            }
        }
        return aggregated_state


class UOCFOGAExpertAlignAggregator(Aggregator):
    # 不带 PISM 的 UOC-FOGA baseline：非专家沿用 FedAvg，专家按 query 梯度方向对齐加权。
    def __init__(self, args, non_expert_method: str):
        self.args = args
        self.non_expert_method = non_expert_method
        self.last_aggregation_metrics = {}
        self._expert_key_cache = None
        self._non_expert_keys = None
        self._cached_key_signature = None
        self.uoc_criterion = nn.CrossEntropyLoss()
        global_query_per_class = int(getattr(args, "uoc_foga_query_per_class", 4))
        global_min_query_samples = int(
            getattr(args, "uoc_foga_min_query_samples_per_expert", 16)
        )
        global_max_samples_per_client_per_class = int(
            getattr(args, "uoc_foga_max_samples_per_client_per_class", 0)
        )
        self.uoc_foga_client_grad_query_per_class = int(getattr(
            args,
            "uoc_foga_client_grad_query_per_class",
            global_query_per_class,
        ))
        self.uoc_foga_client_grad_min_samples_per_expert = int(getattr(
            args,
            "uoc_foga_client_grad_min_samples_per_expert",
            max(1, global_min_query_samples // 2),
        ))
        self.uoc_foga_client_grad_min_classes_per_expert = int(getattr(
            args,
            "uoc_foga_client_grad_min_classes_per_expert",
            1,
        ))
        self.uoc_foga_client_grad_min_expert_token_ratio = float(getattr(
            args,
            "uoc_foga_client_grad_min_expert_token_ratio",
            0.0,
        ))
        self.uoc_foga_client_grad_max_samples_per_client_per_class = int(getattr(
            args,
            "uoc_foga_client_grad_max_samples_per_client_per_class",
            global_max_samples_per_client_per_class,
        ))
        self.uoc_foga_client_grad_fallback_to_random = bool(getattr(
            args,
            "uoc_foga_client_grad_fallback_to_random",
            True,
        ))
        self._validate_client_grad_query_config()

    def _validate_client_grad_query_config(self):
        if self.uoc_foga_client_grad_query_per_class < 1:
            raise ValueError("uoc_foga_client_grad_query_per_class must be >= 1")
        if self.uoc_foga_client_grad_min_samples_per_expert < 1:
            raise ValueError("uoc_foga_client_grad_min_samples_per_expert must be >= 1")
        if self.uoc_foga_client_grad_min_classes_per_expert < 1:
            raise ValueError("uoc_foga_client_grad_min_classes_per_expert must be >= 1")
        if self.uoc_foga_client_grad_min_expert_token_ratio < 0.0:
            raise ValueError("uoc_foga_client_grad_min_expert_token_ratio must be >= 0")
        if self.uoc_foga_client_grad_max_samples_per_client_per_class < 0:
            raise ValueError(
                "uoc_foga_client_grad_max_samples_per_client_per_class must be >= 0"
            )

    def _get_num_classes(self):
        num_classes = getattr(self.args, "num_classes", None)
        if num_classes is not None:
            return int(num_classes)

        data_name = str(getattr(self.args, "data_name", "")).lower()
        if data_name == "cifar100":
            return 100
        if data_name == "cifar10":
            return 10
        return 10

    def _build_key_cache(self, state_keys):
        key_signature = tuple(state_keys)
        if self._cached_key_signature == key_signature:
            return

        expert_key_cache = collections.defaultdict(list)
        non_expert_keys = []
        for key in state_keys:
            parsed_key = parse_expert_parameter_key(key)
            if parsed_key is None:
                non_expert_keys.append(key)
                continue
            expert_key_cache[parsed_key].append(key)

        self._expert_key_cache = {
            key: sorted(keys)
            for key, keys in expert_key_cache.items()
        }
        self._non_expert_keys = non_expert_keys
        self._cached_key_signature = key_signature

    def _get_global_state(self, client_updates, global_model):
        if global_model is None:
            return {
                key: value.detach().cpu().clone()
                for key, value in client_updates[0].items()
            }

        return {
            key: value.detach()
            for key, value in global_model.state_dict().items()
        }

    def _get_model_device(self, global_model):
        if global_model is None:
            return torch.device("cpu")
        try:
            return next(global_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _aggregate_non_expert_keys(self, aggregated_state, client_updates, client_weights):
        normalized_weights = build_client_weights(self.non_expert_method, client_weights)
        for key in self._non_expert_keys:
            first_value = client_updates[0][key].detach().cpu()
            if not torch.is_floating_point(first_value):
                # 非浮点 buffer 沿用旧逻辑：取第一个客户端的值。
                aggregated_state[key] = first_value.clone()
                continue

            aggregated_value = torch.zeros_like(first_value)
            for update, weight in zip(client_updates, normalized_weights):
                aggregated_value += update[key].detach().cpu() * weight
            aggregated_state[key] = aggregated_value

    def _apply_uniform_delta_for_expert(
        self,
        aggregated_state,
        global_state,
        client_updates,
        expert_keys,
    ):
        # uniform_delta 等价于对该 expert 的 local 参数做等权平均。
        for key in expert_keys:
            first_value = client_updates[0][key].detach().cpu()
            global_value = global_state.get(key, first_value).detach().cpu()
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = global_value.clone()
                continue

            valid_values = []
            for client_state in client_updates:
                value = client_state.get(key)
                if value is None or not torch.is_tensor(value):
                    continue
                if not torch.is_floating_point(value):
                    continue
                valid_values.append(value.detach().cpu())

            if not valid_values:
                aggregated_state[key] = global_value.clone()
                continue

            aggregated_state[key] = torch.stack(valid_values, dim=0).mean(dim=0)

    def _apply_weighted_delta_for_expert(
        self,
        aggregated_state,
        global_state,
        client_updates,
        expert_keys,
        weights,
        device,
    ):
        for key in expert_keys:
            if key not in global_state:
                return False
            global_tensor = global_state[key]
            if not torch.is_tensor(global_tensor) or not torch.is_floating_point(global_tensor):
                aggregated_state[key] = global_tensor.detach().cpu().clone()
                continue

            selected_weights = {}
            for client_idx, weight in weights.items():
                if key not in client_updates[client_idx]:
                    return False
                client_tensor = client_updates[client_idx][key]
                if not torch.is_tensor(client_tensor) or not torch.is_floating_point(client_tensor):
                    return False
                selected_weights[client_idx] = float(weight)

            weight_total = sum(selected_weights.values())
            if weight_total <= 0:
                return False

            global_on_device = global_tensor.detach().to(device)
            delta_sum = torch.zeros_like(global_on_device)
            for client_idx, weight in selected_weights.items():
                normalized_weight = weight / weight_total
                client_tensor = client_updates[client_idx][key].detach().to(device)
                delta_sum += normalized_weight * (client_tensor - global_on_device)

            updated_tensor = global_on_device + delta_sum
            target_dtype = global_tensor.detach().cpu().dtype
            aggregated_state[key] = updated_tensor.detach().cpu().to(dtype=target_dtype)

        return True

    def _resolve_uoc_evidences(self, kwargs):
        uoc_evidence = kwargs.get("uoc_evidence")
        if uoc_evidence is not None:
            if isinstance(uoc_evidence, list):
                return uoc_evidence
            if isinstance(uoc_evidence, tuple):
                return list(uoc_evidence)
            return [uoc_evidence]

        client_stats = kwargs.get("client_stats")
        if client_stats is None:
            return None

        if isinstance(client_stats, dict):
            client_stats = list(client_stats.values())

        evidences = []
        for stats in client_stats:
            if not isinstance(stats, dict):
                continue
            evidences.append(stats.get("uoc_evidence_by_layer", {}))
        return evidences

    def _make_empty_expert_metric(self, fallback_reason):
        return {
            "valid_clients": 0,
            "query_size": 0,
            "query_num_classes": 0,
            "query_has_residual": False,
            "score_mean": None,
            "score_min": None,
            "score_max": None,
            "score_pos_frac": None,
            "score_abs_mean": None,
            "score_abs_max": None,
            "score_metric": self._get_score_metric(),
            "weight_max": None,
            "weight_entropy": None,
            "aggregation_weights": None,
            "aggregation_weight_source": None,
            "fallback_reason": fallback_reason,
        }

    def _uniform_weight_dict(self, num_clients):
        if num_clients <= 0:
            return {}
        weight = 1.0 / float(num_clients)
        return {str(client_idx): float(weight) for client_idx in range(num_clients)}

    def _client_weight_dict(self, weights, num_clients=None):
        result = {}
        for client_idx, weight in weights.items():
            result[str(client_idx)] = float(weight)
        return result

    def _mark_uniform_fallback_weights(self, metric, client_updates):
        metric["aggregation_weights"] = self._uniform_weight_dict(len(client_updates))
        metric["aggregation_weight_source"] = "uniform_fallback"

    def _build_expert_aggregation_weights_summary(self, uoc_foga_stats, weight_source):
        by_layer = {}
        for layer_id, layer_stats in uoc_foga_stats.items():
            if not isinstance(layer_stats, dict):
                continue
            layer_weights = {}
            for expert_id, metric in layer_stats.items():
                if not isinstance(metric, dict):
                    continue
                layer_weights[str(expert_id)] = {
                    "weights": metric.get("aggregation_weights") or {},
                    "source": metric.get("aggregation_weight_source"),
                    "fallback_reason": metric.get("fallback_reason"),
                }
            by_layer[str(layer_id)] = layer_weights

        return {
            "weight_source": weight_source,
            "shared_across_experts": False,
            "by_layer": by_layer,
        }

    def _get_score_metric(self):
        score_metric = getattr(self.args, "uoc_foga_score_metric", "cosine")
        if score_metric not in {"cosine", "dot", "grad_dot", "grad_cosine", "delta_consensus"}:
            raise ValueError(f"Unknown UOC-FOGA score metric: {score_metric!r}")
        return score_metric

    def _get_query_select_kwargs(self):
        return {
            "query_select_mode": getattr(
                self.args,
                "uoc_foga_query_select_mode",
                "class_balanced_random",
            ),
            "min_expert_token_ratio": float(
                getattr(self.args, "uoc_foga_min_expert_token_ratio", 0.0)
            ),
            "max_samples_per_client_per_class": int(
                getattr(self.args, "uoc_foga_max_samples_per_client_per_class", 0)
            ),
            "fallback_to_random": bool(
                getattr(self.args, "uoc_foga_query_fallback_to_random", True)
            ),
            # mixed_global_expert 模式专用：只影响 g_query 的 D_query,l,e；
            # g_client,i,l,e 仍然由 build_client_grad_query_for_expert 使用客户端自己的 evidence 构造。
            "global_query_ratio": float(
                getattr(self.args, "uoc_foga_global_query_ratio", 0.5)
            ),
        }

    def _copy_query_stats_to_metric(self, metric, query):
        for key in (
            "query_select_mode",
            "token_ratio_threshold",
            "pool_size_before_filter",
            "pool_size_after_token_ratio_filter",
            "selected_by_entropy",
            "fallback_to_random_used",
            "entropy_missing",
            "expert_token_ratio_mean",
            "expert_token_ratio_min",
            "expert_token_ratio_max",
            "query_entropy_mean",
            "max_samples_per_client_per_class",
            "global_query_ratio",
            "mixed_global_query_size",
            "mixed_expert_query_size",
            "mixed_global_query_num_classes",
            "mixed_expert_query_num_classes",
            "mixed_global_ratio_effective",
        ):
            if key in query:
                metric[key] = query[key]

    def _build_delta_consensus_scores(
        self,
        client_updates,
        global_state,
        expert_keys,
        device,
    ):
        client_scores = {client_idx: None for client_idx in range(len(client_updates))}
        delta_states = {}
        for client_idx, client_state in enumerate(client_updates):
            delta_state = extract_expert_delta_state(
                client_state,
                global_state,
                expert_keys,
                device=device,
            )
            if l2_norm_state(delta_state, device=device) <= 1e-12:
                continue
            delta_states[client_idx] = delta_state

        if len(delta_states) < 2:
            return client_scores, delta_states, [], [], "too_few_delta_clients"

        scores = []
        ref_client_counts = []
        delta_items = list(delta_states.items())
        for client_idx, delta_state in delta_items:
            ref_states = [
                other_delta_state
                for other_client_idx, other_delta_state in delta_items
                if other_client_idx != client_idx
            ]
            ref_delta_state = average_delta_states(ref_states, device=device)
            if ref_delta_state is None:
                continue
            score = cosine_delta_to_delta(
                delta_state,
                ref_delta_state,
                device=device,
            )
            if score is None:
                continue
            client_scores[client_idx] = float(score)
            scores.append(float(score))
            ref_client_counts.append(len(ref_states))

        if not scores:
            return client_scores, delta_states, scores, ref_client_counts, "no_valid_delta_consensus_scores"
        return client_scores, delta_states, scores, ref_client_counts, None

    def _add_delta_consensus_metric_stats(self, metric, scores, ref_client_counts):
        if metric.get("score_metric") != "delta_consensus":
            return metric

        scores = [float(value) for value in scores if value is not None]
        ref_client_counts = [
            float(value) for value in ref_client_counts if value is not None
        ]
        metric["delta_consensus_valid_scores"] = len(scores)
        metric["delta_consensus_positive_frac"] = (
            sum(1 for value in scores if value > 0.0) / len(scores)
            if scores
            else None
        )
        metric["delta_consensus_score_mean"] = self._mean_float_values(scores)
        metric["delta_consensus_score_std"] = self._std_float_values(scores)
        metric["delta_consensus_score_min"] = min(scores) if scores else None
        metric["delta_consensus_score_max"] = max(scores) if scores else None
        metric["delta_consensus_ref_clients_mean"] = self._mean_float_values(
            ref_client_counts
        )
        return metric

    def _get_client_grad_query_kwargs(self):
        """
        构造 g_client,i,l,e 使用的 client-grad query 参数。

        注意：
        - mixed_global_expert 只用于服务端全局 g_query 的 D_query,l,e；
        - g_client,i,l,e 仍然在客户端 i 自己的 expert evidence 上算；
        - 因此当全局 query_select_mode 是 mixed_global_expert 时，
          client-grad query 自动回退到 expert_ratio_entropy。
        """
        query_select_mode = getattr(
            self.args,
            "uoc_foga_query_select_mode",
            "class_balanced_random",
        )

        # mixed_global_expert 是 g_query 的混合全局 query 构造方式，
        # build_client_grad_query_for_expert 不支持也不应该使用它。
        if query_select_mode == "mixed_global_expert":
            client_grad_query_select_mode = "expert_ratio_entropy"
        else:
            client_grad_query_select_mode = query_select_mode

        return {
            "query_per_class": self.uoc_foga_client_grad_query_per_class,
            "min_query_samples": self.uoc_foga_client_grad_min_samples_per_expert,
            "min_classes": self.uoc_foga_client_grad_min_classes_per_expert,
            "query_select_mode": client_grad_query_select_mode,
            "min_expert_token_ratio": self.uoc_foga_client_grad_min_expert_token_ratio,
            "max_samples_per_client_per_class": (
                self.uoc_foga_client_grad_max_samples_per_client_per_class
            ),
            "fallback_to_random": self.uoc_foga_client_grad_fallback_to_random,
        }

    def _get_client_evidence_for_index(self, uoc_evidences, client_idx):
        if not isinstance(uoc_evidences, (list, tuple)):
            return None
        if client_idx < 0 or client_idx >= len(uoc_evidences):
            return None
        return uoc_evidences[client_idx]

    def _compute_client_expert_grad_state(
        self,
        global_model,
        client_query,
        layer_id,
        expert_id,
        expert_param_keys,
        expert_params,
        device,
    ):
        if global_model is None or not isinstance(client_query, dict):
            return None
        hidden = client_query.get("hidden")
        labels = client_query.get("labels")
        if hidden is None or labels is None:
            return None
        if not torch.is_tensor(hidden) or not torch.is_tensor(labels):
            return None
        if hidden.dim() == 0 or labels.dim() == 0 or hidden.size(0) == 0:
            return None
        if not expert_param_keys or not expert_params:
            return None

        hidden = hidden.to(device)
        labels = labels.to(device).long()
        residual = client_query.get("residual")
        if residual is not None:
            if not torch.is_tensor(residual):
                residual = None
            else:
                residual = residual.to(device)
        sample_count = int(client_query.get("sample_count", labels.size(0)))
        if sample_count <= 0:
            return None

        try:
            output = global_model.forward_uoc_from_hidden(
                hidden=hidden,
                residual=residual,
                layer_id=layer_id,
                force_expert_id=expert_id,
                gate_mode=getattr(self.args, "uoc_foga_gate_mode", "one"),
            )
            loss = self.uoc_criterion(output["logits"], labels)
            if not torch.isfinite(loss):
                return None
            grads = torch.autograd.grad(
                loss,
                expert_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return None

        client_grad_state = {
            key: grad.detach()
            for key, grad in zip(expert_param_keys, grads)
            if grad is not None
        }
        if not client_grad_state:
            return None
        return float(loss.detach().cpu().item()), client_grad_state, sample_count

    def _compute_client_grad_dot_score(
        self,
        global_model,
        uoc_evidences,
        client_idx,
        delta_state,
        query_grad_state,
        layer_id,
        expert_id,
        expert_param_keys,
        expert_params,
        device,
        score_metric="grad_dot",
    ):
        client_evidence = self._get_client_evidence_for_index(uoc_evidences, client_idx)
        if client_evidence is None:
            return None, None, None, None, None

        client_query = build_client_grad_query_for_expert(
            client_evidence=client_evidence,
            layer_id=layer_id,
            expert_id=expert_id,
            **self._get_client_grad_query_kwargs(),
        )
        if client_query is None:
            return None, None, None, None, None

        result = self._compute_client_expert_grad_state(
            global_model=global_model,
            client_query=client_query,
            layer_id=layer_id,
            expert_id=expert_id,
            expert_param_keys=expert_param_keys,
            expert_params=expert_params,
            device=device,
        )
        if result is None:
            return None, None, None, None, None

        expert_client_loss, client_grad_state, sample_count = result
        if score_metric == "grad_cosine":
            # grad_cosine 只比较梯度方向，避免 raw grad_dot 被梯度范数主导。
            score = cosine_grad_to_grad(query_grad_state, client_grad_state, device=device)
        else:
            score = dot_grad_to_grad(query_grad_state, client_grad_state, device=device)
        if score is None:
            return None, None, None, None, None
        cos_delta_neg_gclient = delta_to_negative_grad_score(
            delta_state,
            client_grad_state,
            metric="cosine",
            device=device,
        )
        return score, expert_client_loss, sample_count, cos_delta_neg_gclient, client_grad_state

    def _mean_float_values(self, values):
        values = [float(value) for value in values if value is not None]
        if not values:
            return None
        return sum(values) / len(values)

    def _std_float_values(self, values):
        values = [float(value) for value in values if value is not None]
        if not values:
            return None
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        return float(variance ** 0.5)

    def _add_grad_dot_metric_stats(self, metric, scores, client_set_sizes, cos_delta_values):
        score_metric = metric.get("score_metric")
        if score_metric not in {"grad_dot", "grad_cosine"}:
            return metric

        scores = [float(value) for value in scores if value is not None]
        client_set_sizes = [float(value) for value in client_set_sizes if value is not None]
        cos_delta_values = [float(value) for value in cos_delta_values if value is not None]
        prefix = "grad_cosine" if score_metric == "grad_cosine" else "grad_dot"

        metric[f"{prefix}_valid_scores"] = len(scores)
        metric[f"{prefix}_positive_frac"] = (
            sum(1 for value in scores if value > 0.0) / len(scores)
            if scores
            else None
        )
        metric[f"{prefix}_score_mean"] = self._mean_float_values(scores)
        metric[f"{prefix}_score_std"] = self._std_float_values(scores)
        metric[f"{prefix}_score_min"] = min(scores) if scores else None
        metric[f"{prefix}_score_max"] = max(scores) if scores else None
        metric[f"{prefix}_client_set_size_mean"] = self._mean_float_values(client_set_sizes)
        metric[f"{prefix}_client_set_size_min"] = min(client_set_sizes) if client_set_sizes else None
        metric[f"{prefix}_client_set_size_max"] = max(client_set_sizes) if client_set_sizes else None
        metric["cos_delta_neg_gclient_mean"] = self._mean_float_values(cos_delta_values)
        metric["cos_delta_neg_gclient_positive_frac"] = (
            sum(1 for value in cos_delta_values if value > 0.0) / len(cos_delta_values)
            if cos_delta_values
            else None
        )
        return metric

    def _build_expert_params_and_grads(
        self,
        global_model,
        expert_keys,
        query,
        layer_id,
        expert_id,
        device,
        named_parameters=None,
    ):
        if named_parameters is None:
            named_parameters = dict(global_model.named_parameters())
        param_keys = [
            key
            for key in expert_keys
            if key in named_parameters and named_parameters[key].requires_grad
        ]
        if not param_keys:
            return None, None, "missing_expert_params"

        expert_params = [named_parameters[key] for key in param_keys]
        hidden = query["hidden"].to(device)
        labels = query["labels"].to(device).long()
        residual = query.get("residual")
        if residual is not None:
            residual = residual.to(device)

        output = global_model.forward_uoc_from_hidden(
            hidden=hidden,
            residual=residual,
            layer_id=layer_id,
            force_expert_id=expert_id,
            gate_mode=getattr(self.args, "uoc_foga_gate_mode", "one"),
        )
        loss = self.uoc_criterion(output["logits"], labels)
        grads = torch.autograd.grad(
            loss,
            expert_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        grad_state = {
            key: grad.detach()
            for key, grad in zip(param_keys, grads)
            if grad is not None
        }
        if not grad_state or l2_norm_state(grad_state, device=device) <= 1e-12:
            return None, None, "query_grad_too_small"
        return param_keys, grad_state, None

    def _aggregate_expert_with_delta_consensus(
        self,
        aggregated_state,
        global_state,
        client_updates,
        expert_keys,
        metric,
        device,
    ):
        client_scores, _, scores, ref_client_counts, score_fallback = (
            self._build_delta_consensus_scores(
                client_updates=client_updates,
                global_state=global_state,
                expert_keys=expert_keys,
                device=device,
            )
        )
        self._add_delta_consensus_metric_stats(metric, scores, ref_client_counts)
        if score_fallback is not None:
            metric["fallback_reason"] = score_fallback
            self._mark_uniform_fallback_weights(metric, client_updates)
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        score_summary = summarize_scores(client_scores)
        metric.update(score_summary)
        valid_clients = int(score_summary["valid_score_count"])
        metric["valid_clients"] = valid_clients
        min_valid_clients = int(getattr(self.args, "uoc_foga_min_valid_clients", 2))
        if valid_clients < min_valid_clients:
            metric["fallback_reason"] = "too_few_valid_clients"
            self._mark_uniform_fallback_weights(metric, client_updates)
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        weights, weight_fallback = positive_score_to_weights(
            client_scores,
            mode=getattr(self.args, "uoc_foga_score_mode", "relu"),
        )
        if weight_fallback is not None:
            metric["fallback_reason"] = weight_fallback
            self._mark_uniform_fallback_weights(metric, client_updates)
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        if not self._apply_weighted_delta_for_expert(
            aggregated_state,
            global_state,
            client_updates,
            expert_keys,
            weights,
            device,
        ):
            metric["fallback_reason"] = "missing_client_expert_key"
            self._mark_uniform_fallback_weights(metric, client_updates)
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        metric["weight_max"] = max(weights.values()) if weights else None
        metric["weight_entropy"] = expert_weight_entropy(weights)
        metric["aggregation_weights"] = self._client_weight_dict(weights)
        metric["aggregation_weight_source"] = "uoc_foga_score"
        metric["fallback_reason"] = None
        return metric

    def _aggregate_expert_with_uoc_foga(
        self,
        aggregated_state,
        global_state,
        client_updates,
        expert_keys,
        layer_id,
        expert_id,
        global_model,
        uoc_evidences,
        reference_uoc_evidence,
        no_evidence_fallback_reason,
        device,
        named_parameters=None,
    ):
        num_classes = self._get_num_classes()
        metric = self._make_empty_expert_metric(None)

        if metric["score_metric"] == "delta_consensus":
            return self._aggregate_expert_with_delta_consensus(
                aggregated_state=aggregated_state,
                global_state=global_state,
                client_updates=client_updates,
                expert_keys=expert_keys,
                metric=metric,
                device=device,
            )

        if no_evidence_fallback_reason is not None:
            metric["fallback_reason"] = no_evidence_fallback_reason
            self._mark_uniform_fallback_weights(metric, client_updates)
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        query = build_stratified_query_for_expert(
            uoc_evidences,
            layer_id=layer_id,
            expert_id=expert_id,
            num_classes=num_classes,
            query_per_class=getattr(self.args, "uoc_foga_query_per_class", 4),
            min_query_samples=getattr(
                self.args,
                "uoc_foga_min_query_samples_per_expert",
                16,
            ),
            min_query_classes=getattr(self.args, "uoc_foga_min_classes_per_expert", 2),
            use_top1=True,
            seed=getattr(self.args, "seed", None),
            **self._get_query_select_kwargs(),
        )
        metric["query_size"] = int(query.get("query_size", 0))
        metric["query_num_classes"] = int(query.get("query_num_classes", 0))
        metric["query_has_residual"] = bool(query.get("has_residual", False))
        self._copy_query_stats_to_metric(metric, query)

        if query.get("fallback_reason") is not None:
            metric["fallback_reason"] = query["fallback_reason"]
            self._mark_uniform_fallback_weights(metric, client_updates)
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        if global_model is None or not hasattr(global_model, "forward_uoc_from_hidden"):
            metric["fallback_reason"] = "missing_global_model_forward_uoc_from_hidden"
            self._mark_uniform_fallback_weights(metric, client_updates)
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        param_keys, grad_state, grad_fallback = self._build_expert_params_and_grads(
            global_model=global_model,
            expert_keys=expert_keys,
            query=query,
            layer_id=layer_id,
            expert_id=expert_id,
            device=device,
            named_parameters=named_parameters,
        )
        if grad_fallback is not None:
            metric["fallback_reason"] = grad_fallback
            self._mark_uniform_fallback_weights(metric, client_updates)
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        score_metric = metric["score_metric"]
        expert_params = None
        if score_metric in {"grad_dot", "grad_cosine"}:
            if named_parameters is None:
                named_parameters = dict(global_model.named_parameters())
            expert_params = [named_parameters[key] for key in param_keys]

        client_scores = {}
        grad_dot_scores = []
        grad_dot_client_set_sizes = []
        cos_delta_neg_gclient_values = []
        for client_idx, client_state in enumerate(client_updates):
            delta_state = extract_expert_delta_state(
                client_state,
                global_state,
                expert_keys,
                device=device,
            )
            if score_metric in {"grad_dot", "grad_cosine"}:
                score, expert_client_loss, client_set_size, cos_delta_neg_gclient, client_grad_state = self._compute_client_grad_dot_score(
                    global_model=global_model,
                    uoc_evidences=uoc_evidences,
                    client_idx=client_idx,
                    delta_state=delta_state,
                    query_grad_state=grad_state,
                    layer_id=layer_id,
                    expert_id=expert_id,
                    expert_param_keys=param_keys,
                    expert_params=expert_params,
                    device=device,
                    score_metric=score_metric,
                )
                if score is not None:
                    grad_dot_scores.append(float(score))
                    grad_dot_client_set_sizes.append(client_set_size)
                    if cos_delta_neg_gclient is not None:
                        cos_delta_neg_gclient_values.append(float(cos_delta_neg_gclient))
            else:
                score = delta_to_negative_grad_score(
                    delta_state,
                    grad_state,
                    metric=score_metric,
                    device=device,
                )
            client_scores[client_idx] = score

        self._add_grad_dot_metric_stats(
            metric,
            grad_dot_scores,
            grad_dot_client_set_sizes,
            cos_delta_neg_gclient_values,
        )
        score_summary = summarize_scores(client_scores)
        metric.update(score_summary)
        valid_clients = int(score_summary["valid_score_count"])
        metric["valid_clients"] = valid_clients
        min_valid_clients = int(getattr(self.args, "uoc_foga_min_valid_clients", 2))
        if valid_clients < min_valid_clients:
            metric["fallback_reason"] = "too_few_valid_clients"
            self._mark_uniform_fallback_weights(metric, client_updates)
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        weights, score_fallback = positive_score_to_weights(
            client_scores,
            mode=getattr(self.args, "uoc_foga_score_mode", "relu"),
        )
        if score_fallback is not None:
            metric["fallback_reason"] = score_fallback
            self._mark_uniform_fallback_weights(metric, client_updates)
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        if not self._apply_weighted_delta_for_expert(
            aggregated_state,
            global_state,
            client_updates,
            expert_keys,
            weights,
            device,
        ):
            metric["fallback_reason"] = "missing_client_expert_key"
            self._mark_uniform_fallback_weights(metric, client_updates)
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        metric["weight_max"] = max(weights.values()) if weights else None
        metric["weight_entropy"] = expert_weight_entropy(weights)
        metric["aggregation_weights"] = self._client_weight_dict(weights)
        metric["aggregation_weight_source"] = "uoc_foga_score"
        metric["fallback_reason"] = None
        return metric

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        if len(client_updates) == 0:
            raise ValueError("UOCFOGAExpertAlignAggregator requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        self._build_key_cache(client_updates[0].keys())
        global_state = self._get_global_state(client_updates, global_model)
        aggregated_state = {}
        self._aggregate_non_expert_keys(aggregated_state, client_updates, client_weights)

        uoc_evidences = self._resolve_uoc_evidences(kwargs)
        reference_uoc_evidence = kwargs.get("reference_uoc_evidence", None)
        no_evidence_fallback_reason = None
        if not uoc_evidences:
            no_evidence_fallback_reason = "no_uoc_evidence_passed_to_aggregator"

        device = self._get_model_device(global_model)
        # 每轮只构造一次参数字典，避免每个 expert 重复遍历 named_parameters。
        named_parameters = dict(global_model.named_parameters()) if global_model is not None else None
        uoc_foga_stats = {}
        for (layer_id, expert_id), expert_keys in sorted(
            self._expert_key_cache.items(),
            key=lambda item: (int(item[0][0]), int(item[0][1])),
        ):
            layer_stats = uoc_foga_stats.setdefault(str(layer_id), {})
            metric = self._aggregate_expert_with_uoc_foga(
                aggregated_state=aggregated_state,
                global_state=global_state,
                client_updates=client_updates,
                expert_keys=expert_keys,
                layer_id=str(layer_id),
                expert_id=str(expert_id),
                global_model=global_model,
                uoc_evidences=uoc_evidences,
                reference_uoc_evidence=reference_uoc_evidence,
                no_evidence_fallback_reason=no_evidence_fallback_reason,
                device=device,
                named_parameters=named_parameters,
            )
            layer_stats[str(expert_id)] = metric

        self.last_aggregation_metrics = {
            "uoc_foga_stats": uoc_foga_stats,
            "expert_aggregation_weights": self._build_expert_aggregation_weights_summary(
                uoc_foga_stats,
                weight_source="uoc_foga",
            ),
        }
        return collections.OrderedDict(
            (key, aggregated_state[key])
            for key in client_updates[0].keys()
        )



def _to_flat_float_list(values):
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        return [
            float(value)
            for value in values.detach().reshape(-1).float().cpu().tolist()
        ]
    if isinstance(values, (list, tuple)):
        result = []
        for value in values:
            if isinstance(value, torch.Tensor):
                result.extend(_to_flat_float_list(value))
                continue
            try:
                result.append(float(value))
            except (TypeError, ValueError):
                result.append(float("nan"))
        return result
    try:
        return [float(values)]
    except (TypeError, ValueError):
        return []


def _is_finite_float(value):
    try:
        return bool(torch.isfinite(torch.tensor(float(value))).item())
    except (TypeError, ValueError):
        return False


def _safe_numeric_values(values):
    return [
        value
        for value in _to_flat_float_list(values)
        if _is_finite_float(value)
    ]


def _safe_mean(values):
    values = _safe_numeric_values(values)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _safe_std(values):
    values = _safe_numeric_values(values)
    if len(values) < 2:
        return float("nan")
    tensor = torch.tensor(values, dtype=torch.float64)
    centered = tensor - tensor.mean()
    return float(torch.sqrt(torch.mean(centered * centered)).item())


def _safe_zscore(values, eps=1e-8, device=None, dtype=torch.float32):
    x = torch.as_tensor(values, device=device, dtype=dtype).reshape(-1)
    if x.numel() == 0:
        return x

    finite_mask = torch.isfinite(x)
    if not torch.any(finite_mask):
        return torch.zeros_like(x)

    finite_values = x[finite_mask]
    fill_value = finite_values.mean()
    x = torch.where(finite_mask, x, fill_value)
    if x.numel() < 2:
        return torch.zeros_like(x)

    std = x.std(unbiased=False)
    if not torch.isfinite(std) or std.item() < eps:
        return torch.zeros_like(x)

    z = (x - x.mean()) / (std + eps)
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_rank_norm_desc(values, device=None, dtype=torch.float32):
    values = torch.as_tensor(values, device=device, dtype=dtype).reshape(-1)
    if values.numel() == 0:
        return values
    if values.numel() == 1:
        return torch.ones_like(values)

    values = torch.nan_to_num(values, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf"))
    ranked_indices = torch.argsort(values, descending=True, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    rank_values = torch.arange(1, values.numel() + 1, device=values.device, dtype=torch.float32)
    ranks[ranked_indices] = rank_values
    return 1.0 - (ranks - 1.0) / max(float(values.numel() - 1), 1.0)


def _safe_pearson(x, y):
    x_values = _to_flat_float_list(x)
    y_values = _to_flat_float_list(y)
    pairs = [
        (x_value, y_value)
        for x_value, y_value in zip(x_values, y_values)
        if _is_finite_float(x_value) and _is_finite_float(y_value)
    ]
    if len(pairs) < 2:
        return float("nan")

    x_tensor = torch.tensor([pair[0] for pair in pairs], dtype=torch.float64)
    y_tensor = torch.tensor([pair[1] for pair in pairs], dtype=torch.float64)
    x_centered = x_tensor - x_tensor.mean()
    y_centered = y_tensor - y_tensor.mean()
    denom = torch.sqrt(torch.sum(x_centered * x_centered) * torch.sum(y_centered * y_centered))
    if not torch.isfinite(denom) or denom.item() <= 0.0:
        return float("nan")
    return float(torch.sum(x_centered * y_centered).item() / denom.item())


def _rank_desc_index(values, index):
    values = _to_flat_float_list(values)
    try:
        target_index = int(index)
    except (TypeError, ValueError):
        return float("nan")
    valid_pairs = [
        (idx, value)
        for idx, value in enumerate(values)
        if _is_finite_float(value)
    ]
    if not valid_pairs or target_index not in {idx for idx, _ in valid_pairs}:
        return float("nan")
    ranked = sorted(valid_pairs, key=lambda item: (-item[1], item[0]))
    for rank, (idx, _) in enumerate(ranked, start=1):
        if idx == target_index:
            return float(rank)
    return float("nan")


def _build_pism_alignment_diagnostics(scores, weights):
    score_values = _to_flat_float_list(scores)
    weight_values = _to_flat_float_list(weights)
    pairs = [
        (idx, score, weight)
        for idx, (score, weight) in enumerate(zip(score_values, weight_values))
        if _is_finite_float(score) and _is_finite_float(weight)
    ]
    diagnostics = {
        "score_std": _safe_std(score_values),
        "score_pos_frac": float("nan"),
        "weight_score_corr": _safe_pearson(score_values, weight_values),
        "pism_top_client_score_rank": float("nan"),
        "pism_top_client_score_value": float("nan"),
        "foga_top_client_pism_weight": float("nan"),
    }
    valid_scores = _safe_numeric_values(score_values)
    if valid_scores:
        diagnostics["score_pos_frac"] = float(
            sum(1 for score in valid_scores if score > 0.0) / len(valid_scores)
        )
    if not pairs:
        return diagnostics

    pism_top_index, pism_top_score, _ = max(pairs, key=lambda item: (item[2], -item[0]))
    _, _, foga_top_weight = max(pairs, key=lambda item: (item[1], -item[0]))
    diagnostics["pism_top_client_score_rank"] = _rank_desc_index(score_values, pism_top_index)
    diagnostics["pism_top_client_score_value"] = float(pism_top_score)
    diagnostics["foga_top_client_pism_weight"] = float(foga_top_weight)
    return diagnostics


def _rank_desc_list(values):
    values = _to_flat_float_list(values)
    valid_pairs = [
        (idx, value)
        for idx, value in enumerate(values)
        if _is_finite_float(value)
    ]
    ranks = [float("nan") for _ in values]
    ranked = sorted(valid_pairs, key=lambda item: (-item[1], item[0]))
    for rank, (idx, _) in enumerate(ranked, start=1):
        ranks[idx] = int(rank)
    return ranks


def _safe_float_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(value):
        return value
    return None


def _safe_rank_value_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return int(value)

class UOCFOGAPISMExpertAlignAggregator(UOCFOGAExpertAlignAggregator):
    # PISM 版 UOC-FOGA：用 DeepSets 元网络从 client/expert 特征生成专家聚合权重。
    def __init__(self, args):
        super(UOCFOGAPISMExpertAlignAggregator, self).__init__(
            args=args,
            non_expert_method=args.non_expert_agg_method,
        )
        self.score_metric = self._get_score_metric()
        if self.score_metric == "grad_cosine":
            # grad_cosine 版删掉绝对 usage 以及离散 consensus rank/pos flag，
            # 只保留 3 维 PISM 输入：expert_loss_z / usage_ratio_z / consensus_grad_cos。
            self.pism_input_dim = 3
        else:
            self.pism_input_dim = int(getattr(args, "uoc_foga_pism_input_dim", 5))
        if self.score_metric == "grad_cosine":
            pass
        elif self.score_metric == "grad_dot":
            if self.pism_input_dim != 2:
                raise ValueError(
                    "grad_dot PISM requires uoc_foga_pism_input_dim: 2"
                )
        elif self.score_metric in {"cosine", "delta_consensus"}:
            if self.pism_input_dim not in {3, 5}:
                raise ValueError(
                    f"{self.score_metric} PISM requires uoc_foga_pism_input_dim: 3 or 5"
                )
        elif self.pism_input_dim != 5:
            raise ValueError(
                "当前版本非 grad_dot PISM features 是 5 维，需要设置 uoc_foga_pism_input_dim: 5."
            )
        self.pism_hidden_size = int(getattr(args, "uoc_foga_pism_hidden_size", 64))
        self.pism_dropout = float(getattr(args, "uoc_foga_pism_dropout", 0.0))
        self.pism_lr = float(getattr(args, "uoc_foga_pism_lr", 1e-3))
        self.pism_tau = float(getattr(args, "uoc_foga_pism_tau", 1.0))
        self.pism_tau_schedule = str(
            getattr(args, "uoc_foga_pism_tau_schedule", "constant")
        ).lower()
        self.pism_tau_init = float(
            getattr(args, "uoc_foga_pism_tau_init", self.pism_tau)
        )
        self.pism_tau_min = float(
            getattr(args, "uoc_foga_pism_tau_min", self.pism_tau)
        )
        self.pism_tau_decay = float(
            getattr(args, "uoc_foga_pism_tau_decay", 1.0)
        )
        self._validate_pism_tau_schedule()
        self.pism_renorm_inputs = bool(getattr(args, "uoc_foga_pism_renorm_inputs", True))
        self.pism_min_clients = int(getattr(args, "uoc_foga_pism_min_clients", 2))
        self.uoc_foga_pism_min_weight_factor = float(
            getattr(args, "uoc_foga_pism_min_weight_factor", 0.0)
        )
        self.uoc_foga_pism_fairness_blend = float(
            getattr(args, "uoc_foga_pism_fairness_blend", 0.0)
        )
        # 诊断 A/C：reference gradient / reference step 只记录日志，不参与训练和聚合。
        self.uoc_foga_ref_grad_diag_enabled = bool(
            getattr(args, "uoc_foga_ref_grad_diag_enabled", False)
        )
        self.uoc_foga_ref_step_diag_enabled = bool(
            getattr(args, "uoc_foga_ref_step_diag_enabled", False)
        )
        self.uoc_foga_ref_step_lr = float(
            getattr(args, "uoc_foga_ref_step_lr", 0.01)
        )
        if self.uoc_foga_ref_step_lr <= 0.0:
            raise ValueError("uoc_foga_ref_step_lr must be > 0")
        # PISM client 顺序对齐 debug：只记录日志，不改变训练、score、loss 或权重。
        self.uoc_foga_pism_debug_alignment = bool(
            getattr(args, "uoc_foga_pism_debug_alignment", False)
        )
        self.uoc_foga_pism_debug_alignment_every = max(
            1, int(getattr(args, "uoc_foga_pism_debug_alignment_every", 1))
        )
        self.uoc_foga_pism_debug_alignment_max_records = max(
            0, int(getattr(args, "uoc_foga_pism_debug_alignment_max_records", 2))
        )
        self.uoc_foga_pism_debug_alignment_topk = max(
            1, int(getattr(args, "uoc_foga_pism_debug_alignment_topk", 10))
        )
        if self.uoc_foga_pism_min_weight_factor < 0.0:
            raise ValueError("uoc_foga_pism_min_weight_factor must be >= 0")
        if not 0.0 <= self.uoc_foga_pism_fairness_blend <= 1.0:
            raise ValueError("uoc_foga_pism_fairness_blend must be in [0, 1]")
        # meta_steps 表示每轮同一批 PISM records 上的 optimizer step 次数，不会重复构造 query / g_query。
        raw_meta_steps = getattr(args, "uoc_foga_pism_meta_steps", 1)
        try:
            self.pism_meta_steps = int(raw_meta_steps)
        except (TypeError, ValueError):
            raise ValueError("uoc_foga_pism_meta_steps must be a positive integer")
        if isinstance(raw_meta_steps, bool):
            raise ValueError("uoc_foga_pism_meta_steps must be a positive integer")
        if isinstance(raw_meta_steps, float) and not raw_meta_steps.is_integer():
            raise ValueError("uoc_foga_pism_meta_steps must be a positive integer")
        if isinstance(raw_meta_steps, str) and raw_meta_steps.strip() != str(self.pism_meta_steps):
            raise ValueError("uoc_foga_pism_meta_steps must be a positive integer")
        if self.pism_meta_steps <= 0:
            raise ValueError("uoc_foga_pism_meta_steps must be a positive integer")
        self.pism_update_steps = 0
        self.meta_net = ExpertPISM(
            input_dim=self.pism_input_dim,
            hidden_size=self.pism_hidden_size,
            dropout=self.pism_dropout,
        )
        self.meta_optimizer = torch.optim.Adam(
            self.meta_net.parameters(),
            lr=self.pism_lr,
        )

    def _generic_pism_feature_names(self):
        return [
            "client_loss",
            "log1p_expert_usage",
            "expert_usage_ratio",
            "log1p_delta_norm",
            "log1p_delta_norm_per_sqrt_usage",
        ]

    def _pism_input_names_for_config(self):
        if self.score_metric == "grad_cosine":
            return [
                "expert_loss_z",
                "usage_ratio_z",
                "consensus_grad_cos",
            ]
        if self.score_metric == "grad_dot":
            return [
                "expert_client_loss",
                "log1p_client_grad_sample_count",
            ]
        feature_names = self._generic_pism_feature_names()
        if self.score_metric in {"cosine", "delta_consensus"} and self.pism_input_dim == 3:
            return [feature_names[0], feature_names[1], feature_names[3]]
        return feature_names

    def _select_pism_features_for_config(self, features):
        if self.score_metric in {"cosine", "delta_consensus"} and self.pism_input_dim == 3:
            # 3 维保持旧版语义：[client_loss, log1p(usage), log1p(delta_norm)]。
            return features[..., [0, 1, 3]]
        return features

    def _postprocess_pism_weights(self, weights, fairness_values=None):
        weights = weights.float()
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        if weights.numel() == 0:
            return weights

        weight_sum = weights.sum()
        if not torch.isfinite(weight_sum) or weight_sum.item() <= 0.0:
            weights = torch.ones_like(weights) / float(weights.numel())
        else:
            weights = weights / weight_sum

        num_clients = weights.numel()
        if self.uoc_foga_pism_min_weight_factor > 0.0:
            min_weight = self.uoc_foga_pism_min_weight_factor * (1.0 / float(num_clients))
            weights = torch.clamp(weights, min=min_weight)
            weights = weights / weights.sum()

        blend = float(self.uoc_foga_pism_fairness_blend)
        if blend > 0.0 and fairness_values is not None:
            fairness = fairness_values.to(weights.device).float()
            fairness = torch.nan_to_num(fairness, nan=0.0, posinf=0.0, neginf=0.0)
            fairness = torch.clamp(fairness, min=0.0)
            fairness_sum = fairness.sum()
            if (
                fairness.numel() == num_clients
                and torch.isfinite(fairness_sum)
                and fairness_sum.item() > 0.0
            ):
                fairness = fairness / fairness_sum
                weights = (1.0 - blend) * weights + blend * fairness
                weights = weights / weights.sum()

        weight_sum = weights.sum()
        if not torch.isfinite(weight_sum) or weight_sum.item() <= 0.0:
            return torch.ones_like(weights) / float(weights.numel())
        return weights / weight_sum

    def _validate_pism_tau_schedule(self):
        if self.pism_tau_schedule not in {"constant", "source_exp"}:
            raise ValueError(
                "uoc_foga_pism_tau_schedule must be 'constant' or 'source_exp'"
            )
        if self.pism_tau <= 0.0:
            raise ValueError("uoc_foga_pism_tau must be > 0")
        if self.pism_tau_init <= 0.0:
            raise ValueError("uoc_foga_pism_tau_init must be > 0")
        if self.pism_tau_min <= 0.0:
            raise ValueError("uoc_foga_pism_tau_min must be > 0")
        if self.pism_tau_decay <= 0.0 or self.pism_tau_decay > 1.0:
            raise ValueError("uoc_foga_pism_tau_decay must be in (0, 1]")
        if self.pism_tau_min > self.pism_tau_init:
            raise ValueError("uoc_foga_pism_tau_min must be <= uoc_foga_pism_tau_init")

    def _get_current_pism_tau(self, round_index=None):
        # tau 越大权重越平滑，tau 越小权重越尖锐；退火表示前期平滑、后期逐渐尖锐。
        if self.pism_tau_schedule == "constant":
            return float(self.pism_tau)

        if self.pism_tau_schedule == "source_exp":
            if round_index is None:
                round_index = int(self.pism_update_steps) + 1
            round_index = max(1, int(round_index))
            exponent = round_index - 1
            tau = self.pism_tau_init * (self.pism_tau_decay ** exponent)
            return float(max(self.pism_tau_min, tau))

        raise ValueError(f"Unknown PISM tau schedule: {self.pism_tau_schedule!r}")

    def _validate_checkpoint_input_dim(self, pism_config):
        saved_input_dim = pism_config.get("input_dim")
        if saved_input_dim is None:
            return
        if int(saved_input_dim) != self.pism_input_dim:
            raise ValueError(
                "PISM input_dim mismatch. Old checkpoint is incompatible after changing PISM features. Please set resume=false or use a new run_name."
            )

    def _get_checkpoint_tau_schedule_config(self, pism_config):
        saved_tau = float(pism_config.get("tau", self.pism_tau))
        return {
            "tau_schedule": str(pism_config.get("tau_schedule", "constant")).lower(),
            "tau_init": float(pism_config.get("tau_init", saved_tau)),
            "tau_min": float(pism_config.get("tau_min", saved_tau)),
            "tau_decay": float(pism_config.get("tau_decay", 1.0)),
        }

    def _validate_checkpoint_tau_schedule(self, pism_config):
        saved_config = self._get_checkpoint_tau_schedule_config(pism_config)
        current_config = {
            "tau_schedule": self.pism_tau_schedule,
            "tau_init": self.pism_tau_init,
            "tau_min": self.pism_tau_min,
            "tau_decay": self.pism_tau_decay,
        }
        mismatch = saved_config["tau_schedule"] != current_config["tau_schedule"]
        for key in ("tau_init", "tau_min", "tau_decay"):
            if abs(float(saved_config[key]) - float(current_config[key])) > 1e-12:
                mismatch = True
        if mismatch:
            raise ValueError(
                "PISM tau schedule mismatch. Please set resume=false when changing tau schedule."
            )

    def _validate_checkpoint_meta_steps(self, pism_config):
        saved_meta_steps = int(pism_config.get("meta_steps", 1))
        if saved_meta_steps != self.pism_meta_steps:
            raise ValueError(
                "PISM meta_steps mismatch. Please set resume=false when changing uoc_foga_pism_meta_steps."
            )

    def get_checkpoint_state(self):
        return {
            "type": "uoc_foga_pism_expert_align",
            "meta_net": self.meta_net.state_dict(),
            "meta_optimizer": self.meta_optimizer.state_dict(),
            "pism_update_steps": int(self.pism_update_steps),
            "pism_config": {
                "input_dim": self.pism_input_dim,
                "hidden_size": self.pism_hidden_size,
                "dropout": self.pism_dropout,
                "lr": self.pism_lr,
                "tau": self.pism_tau,
                "tau_schedule": self.pism_tau_schedule,
                "tau_init": self.pism_tau_init,
                "tau_min": self.pism_tau_min,
                "tau_decay": self.pism_tau_decay,
                "meta_steps": self.pism_meta_steps,
                "renorm_inputs": self.pism_renorm_inputs,
                "min_clients": self.pism_min_clients,
            },
        }

    def load_checkpoint_state(self, state, map_location=None):
        if not state:
            return
        if not isinstance(state, dict):
            return
        if state.get("type") != "uoc_foga_pism_expert_align":
            print("aggregator checkpoint type mismatch")
            return

        pism_config = state.get("pism_config", {})
        if not isinstance(pism_config, dict):
            pism_config = {}
        self._validate_checkpoint_input_dim(pism_config)
        self._validate_checkpoint_tau_schedule(pism_config)
        self._validate_checkpoint_meta_steps(pism_config)

        if "meta_net" in state:
            self.meta_net.load_state_dict(state["meta_net"])
        if "meta_optimizer" in state:
            self.meta_optimizer.load_state_dict(state["meta_optimizer"])
        self.pism_update_steps = int(state.get("pism_update_steps", 0))

    def _move_meta_optimizer_state_to_device(self, device):
        # resume 后 optimizer state 可能还在 CPU，聚合前搬到 meta_net 同设备。
        for optimizer_state in self.meta_optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(device)

    def _add_pism_metric_defaults(self, metric):
        metric.update({
            "pism_used": False,
            "pism_meta_loss": None,
            "pism_weight_max": None,
            "pism_weight_entropy": None,
            "pism_input_mean": None,
            "pism_input_std": None,
            "pism_input_names": self._pism_input_names_for_config(),
            "pism_fallback_reason": None,
        })
        return metric

    def _get_indexed_value(self, value, index):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.dim() == 0 or index >= value.size(0):
                return None
            return float(value[index].detach().cpu().item())
        if isinstance(value, (list, tuple)):
            if index >= len(value):
                return None
            return float(value[index])
        if isinstance(value, dict):
            if index in value:
                return float(value[index])
            key = str(index)
            if key in value:
                return float(value[key])
        return None

    def _get_expert_usage_from_stats(self, client_stat, layer_id, expert_id):
        if not isinstance(client_stat, dict):
            return 0.0

        expert_index = int(expert_id)
        layer_key = str(layer_id)
        activations_by_layer = client_stat.get("expert_activations_by_layer")
        if isinstance(activations_by_layer, dict):
            value = activations_by_layer.get(layer_key, activations_by_layer.get(int(layer_id), None))
            indexed_value = self._get_indexed_value(value, expert_index)
            if indexed_value is not None:
                return indexed_value

        stats_by_layer = client_stat.get("expert_stats_by_layer")
        if isinstance(stats_by_layer, dict):
            layer_stats = stats_by_layer.get(layer_key, stats_by_layer.get(int(layer_id), None))
            if isinstance(layer_stats, dict):
                indexed_value = self._get_indexed_value(
                    layer_stats.get("expert_activations"),
                    expert_index,
                )
                if indexed_value is not None:
                    return indexed_value

        indexed_value = self._get_indexed_value(
            client_stat.get("expert_activations"),
            expert_index,
        )
        if indexed_value is not None:
            return indexed_value
        return 0.0

    def _sum_expert_usage_value(self, value):
        try:
            if value is None:
                return 0.0
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return 0.0
                return float(value.detach().cpu().float().sum().item())
            if isinstance(value, dict):
                total = 0.0
                for item in value.values():
                    total += self._sum_expert_usage_value(item)
                return float(total)
            if isinstance(value, (list, tuple)):
                total = 0.0
                for item in value:
                    total += self._sum_expert_usage_value(item)
                return float(total)
            return float(value)
        except (TypeError, ValueError, RuntimeError):
            return 0.0

    def _get_total_expert_usage_from_stats(self, client_stat, layer_id):
        if not isinstance(client_stat, dict):
            return 0.0

        try:
            layer_key = str(layer_id)
            try:
                layer_index = int(layer_id)
            except (TypeError, ValueError):
                layer_index = None

            activations_by_layer = client_stat.get("expert_activations_by_layer")
            if isinstance(activations_by_layer, dict):
                value = activations_by_layer.get(layer_key)
                if value is None and layer_index is not None:
                    value = activations_by_layer.get(layer_index)
                if value is not None:
                    return self._sum_expert_usage_value(value)

            stats_by_layer = client_stat.get("expert_stats_by_layer")
            if isinstance(stats_by_layer, dict):
                layer_stats = stats_by_layer.get(layer_key)
                if layer_stats is None and layer_index is not None:
                    layer_stats = stats_by_layer.get(layer_index)
                if isinstance(layer_stats, dict) and "expert_activations" in layer_stats:
                    return self._sum_expert_usage_value(layer_stats.get("expert_activations"))

            if "expert_activations" in client_stat:
                return self._sum_expert_usage_value(client_stat.get("expert_activations"))
        except (TypeError, ValueError, RuntimeError):
            return 0.0
        return 0.0

    def _get_client_loss_from_stats(self, client_stat):
        if not isinstance(client_stat, dict):
            return 0.0
        return float(client_stat.get("client_loss", client_stat.get("train_loss", 0.0)))

    def _get_expert_loss_from_stats(self, client_stat, layer_id, expert_id):
        if not isinstance(client_stat, dict):
            return None

        expert_index = int(expert_id)
        layer_key = str(layer_id)
        loss_by_layer = client_stat.get("expert_loss_by_layer")
        if not isinstance(loss_by_layer, dict):
            return None

        layer_loss = loss_by_layer.get(layer_key, loss_by_layer.get(int(layer_id), None))
        if not isinstance(layer_loss, dict):
            return None

        loss_sum = self._get_indexed_value(layer_loss.get("loss_sum"), expert_index)
        loss_count = self._get_indexed_value(layer_loss.get("loss_count"), expert_index)
        if loss_sum is None or loss_count is None:
            return None
        if loss_count <= 0.0:
            return None
        value = float(loss_sum) / max(float(loss_count), 1.0)
        if not _safe_numeric_values([value]):
            return None
        return value

    def _build_consensus_grad_cos_features(self, client_grad_states, device):
        if len(client_grad_states) < 2:
            return torch.zeros(len(client_grad_states), device=device, dtype=torch.float32)

        consensus_values = []
        for idx, client_grad_state in enumerate(client_grad_states):
            ref_states = [
                other_grad_state
                for other_idx, other_grad_state in enumerate(client_grad_states)
                if other_idx != idx
            ]
            ref_grad_state = average_delta_states(ref_states, device=device)
            if ref_grad_state is None:
                consensus_values.append(0.0)
                continue
            score = cosine_grad_to_grad(
                ref_grad_state,
                client_grad_state,
                device=device,
            )
            consensus_values.append(float(score) if score is not None else 0.0)
        return torch.as_tensor(consensus_values, device=device, dtype=torch.float32)

    def _build_grad_cosine_pism_features_for_expert(
        self,
        client_stats,
        valid_client_ids,
        expert_usages,
        total_layer_usages,
        client_grad_states,
        layer_id,
        expert_id,
        device,
    ):
        expert_loss_values = []
        for client_idx in valid_client_ids:
            client_stat = client_stats[client_idx] if client_idx < len(client_stats) else {}
            expert_loss = self._get_expert_loss_from_stats(client_stat, layer_id, expert_id)
            expert_loss_values.append(
                float(expert_loss) if expert_loss is not None else float("nan")
            )

        expert_loss_z = _safe_zscore(expert_loss_values, device=device)
        expert_usage = torch.as_tensor(expert_usages, device=device, dtype=torch.float32).reshape(-1)
        total_layer_usage = torch.as_tensor(total_layer_usages, device=device, dtype=torch.float32).reshape(-1)
        expert_usage = torch.nan_to_num(expert_usage, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        total_layer_usage = torch.nan_to_num(total_layer_usage, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        # 删掉 log_usage_z：绝对 expert usage 容易成为客户端样本量代理，
        # 只保留 usage_ratio_z 表示该 expert 在当前客户端当前层中的相对重要性。
        usage_ratio = torch.where(
            total_layer_usage > 0.0,
            expert_usage / total_layer_usage.clamp_min(1e-8),
            torch.zeros_like(expert_usage),
        )
        usage_ratio_z = _safe_zscore(usage_ratio, device=device)

        consensus_grad_cos = self._build_consensus_grad_cos_features(client_grad_states, device)
        # consensus_grad_rank_norm / consensus_grad_pos_flag 只作为旧诊断含义保留，不再进入 PISM 输入。
        # 这一步用于验证后期 PISM 是否被离散 consensus 排名/正负标记带偏。
        consensus_grad_pos_flag = (consensus_grad_cos > 0.0).float()

        features = torch.stack(
            [
                expert_loss_z,
                usage_ratio_z,
                consensus_grad_cos,
            ],
            dim=-1,
        )
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        feature_diag = {
            "consensus_grad_cos_mean": _safe_mean(consensus_grad_cos),
            "consensus_grad_pos_frac": _safe_mean(consensus_grad_pos_flag),
            "expert_loss_z_std": _safe_std(expert_loss_z),
        }
        return features, feature_diag

    def _compute_query_ref_cos_diag(
        self,
        global_model,
        reference_uoc_evidence,
        query_grad_state,
        expert_keys,
        layer_id,
        expert_id,
        device,
        named_parameters=None,
    ):
        """诊断 A：比较当前 g_query 与 fixed/global balanced reference gradient。"""
        if not self.uoc_foga_ref_grad_diag_enabled:
            return None, None
        if reference_uoc_evidence is None or global_model is None:
            return None, None
        if query_grad_state is None:
            return None, None

        ref_query = build_reference_query_for_layer(
            reference_uoc_evidence=reference_uoc_evidence,
            layer_id=layer_id,
            num_classes=self._get_num_classes(),
        )
        if ref_query is None:
            return None, None

        param_keys, ref_grad_state, ref_fallback = self._build_expert_params_and_grads(
            global_model=global_model,
            expert_keys=expert_keys,
            query=ref_query,
            layer_id=layer_id,
            expert_id=expert_id,
            device=device,
            named_parameters=named_parameters,
        )
        if ref_fallback is not None or ref_grad_state is None:
            return None, ref_query

        query_ref_cos = cosine_grad_to_grad(
            query_grad_state,
            ref_grad_state,
            device=device,
        )
        return query_ref_cos, ref_query

    def _compute_reference_loss_for_query(
        self,
        global_model,
        ref_query,
        layer_id,
        expert_id,
        device,
    ):
        if global_model is None or not isinstance(ref_query, dict):
            return None
        hidden = ref_query.get("hidden")
        labels = ref_query.get("labels")
        if hidden is None or labels is None:
            return None
        if not torch.is_tensor(hidden) or not torch.is_tensor(labels):
            return None
        if hidden.dim() == 0 or labels.dim() == 0 or hidden.size(0) == 0:
            return None

        hidden = hidden.to(device)
        labels = labels.to(device).long()
        residual = ref_query.get("residual")
        if residual is not None:
            residual = residual.to(device) if torch.is_tensor(residual) else None

        was_training = global_model.training
        try:
            global_model.eval()
            with torch.no_grad():
                output = global_model.forward_uoc_from_hidden(
                    hidden=hidden,
                    residual=residual,
                    layer_id=layer_id,
                    force_expert_id=expert_id,
                    gate_mode=getattr(self.args, "uoc_foga_gate_mode", "one"),
                )
                loss = self.uoc_criterion(output["logits"], labels)
            if not torch.isfinite(loss):
                return None
            return float(loss.detach().cpu().item())
        except (KeyError, RuntimeError, TypeError, ValueError):
            return None
        finally:
            global_model.train(was_training)

    def _run_ref_step_diag_for_client_grad(
        self,
        global_model,
        ref_query,
        layer_id,
        expert_id,
        expert_param_keys,
        client_grad_state,
        device,
    ):
        """诊断 C：临时小步 expert 更新，观察 balanced reference loss 是否下降。"""
        if not self.uoc_foga_ref_step_diag_enabled:
            return None
        if global_model is None or ref_query is None or client_grad_state is None:
            return None
        if not expert_param_keys:
            return None

        named_parameters = dict(global_model.named_parameters())
        params = []
        grads = []
        for key in expert_param_keys:
            param = named_parameters.get(key)
            grad = client_grad_state.get(key) if isinstance(client_grad_state, dict) else None
            if param is None or grad is None:
                continue
            if not torch.is_tensor(grad) or tuple(param.shape) != tuple(grad.shape):
                continue
            params.append(param)
            grads.append(grad.to(param.device))
        if not params:
            return None

        loss_before = self._compute_reference_loss_for_query(
            global_model=global_model,
            ref_query=ref_query,
            layer_id=layer_id,
            expert_id=expert_id,
            device=device,
        )
        if loss_before is None:
            return None

        originals = [param.detach().clone() for param in params]
        try:
            with torch.no_grad():
                for param, grad in zip(params, grads):
                    param.add_(grad, alpha=-float(self.uoc_foga_ref_step_lr))
            loss_after = self._compute_reference_loss_for_query(
                global_model=global_model,
                ref_query=ref_query,
                layer_id=layer_id,
                expert_id=expert_id,
                device=device,
            )
        finally:
            with torch.no_grad():
                for param, original in zip(params, originals):
                    param.copy_(original)

        if loss_after is None:
            return None
        return float(loss_after - loss_before)

    def _normalize_client_stats(self, client_stats, client_count):
        if client_stats is None:
            return [{} for _ in range(client_count)]
        if isinstance(client_stats, dict):
            normalized = []
            for client_idx in range(client_count):
                normalized.append(
                    client_stats.get(client_idx, client_stats.get(str(client_idx), {}))
                )
            return normalized
        normalized = list(client_stats)
        if len(normalized) < client_count:
            normalized.extend({} for _ in range(client_count - len(normalized)))
        return normalized[:client_count]

    def _fallback_expert(self, aggregated_state, global_state, client_updates, expert_keys, metric, reason):
        metric["fallback_reason"] = reason
        metric["pism_fallback_reason"] = reason
        self._mark_uniform_fallback_weights(metric, client_updates)
        self._apply_uniform_delta_for_expert(
            aggregated_state,
            global_state,
            client_updates,
            expert_keys,
        )
        return None

    def _build_delta_consensus_pism_record_for_expert(
        self,
        aggregated_state,
        global_state,
        client_updates,
        client_stats,
        client_weights,
        expert_keys,
        layer_id,
        expert_id,
        metric,
        device,
    ):
        client_scores, delta_states, scores_for_stats, ref_client_counts, score_fallback = (
            self._build_delta_consensus_scores(
                client_updates=client_updates,
                global_state=global_state,
                expert_keys=expert_keys,
                device=device,
            )
        )
        self._add_delta_consensus_metric_stats(
            metric,
            scores_for_stats,
            ref_client_counts,
        )
        if score_fallback is not None:
            return metric, self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
                metric,
                score_fallback,
            )

        score_summary = summarize_scores(client_scores)
        metric.update(score_summary)
        valid_clients = int(score_summary["valid_score_count"])
        metric["valid_clients"] = valid_clients
        if valid_clients < 2 or valid_clients < self.pism_min_clients:
            return metric, self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
                metric,
                "too_few_pism_clients",
            )

        valid_client_ids = []
        scores = []
        client_losses = []
        expert_usages = []
        total_layer_usages = []
        delta_norms = []
        for client_idx, score in client_scores.items():
            if score is None:
                continue
            delta_state = delta_states.get(client_idx)
            if delta_state is None:
                continue

            valid_client_ids.append(client_idx)
            scores.append(float(score))
            client_stat = client_stats[client_idx] if client_idx < len(client_stats) else {}
            expert_usage = self._get_expert_usage_from_stats(client_stat, layer_id, expert_id)
            total_layer_usage = self._get_total_expert_usage_from_stats(client_stat, layer_id)
            if total_layer_usage < expert_usage:
                total_layer_usage = expert_usage
            client_losses.append(self._get_client_loss_from_stats(client_stat))
            expert_usages.append(expert_usage)
            total_layer_usages.append(total_layer_usage)
            delta_norms.append(l2_norm_state(delta_state, device=device))

        if len(valid_client_ids) < 2 or len(valid_client_ids) < self.pism_min_clients:
            return metric, self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
                metric,
                "too_few_pism_clients",
            )

        features = build_pism_feature_tensor(
            client_loss=client_losses,
            expert_usage=expert_usages,
            delta_norm=delta_norms,
            total_layer_usage=total_layer_usages,
            device=device,
        )
        features = self._select_pism_features_for_config(features)
        metric["pism_input_names"] = self._pism_input_names_for_config()
        if self.pism_renorm_inputs:
            features = normalize_pism_inputs(features)
        if not torch.isfinite(features).all():
            return metric, self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
                metric,
                "pism_weights_nan",
            )

        metric["pism_input_mean"] = [
            float(value) for value in features.detach().mean(dim=0).cpu().tolist()
        ]
        metric["pism_input_std"] = [
            float(value) for value in features.detach().std(dim=0, unbiased=False).cpu().tolist()
        ]
        # 诊断 B：FOGA score 是否和客户端样本量相关。
        metric["score_sample_corr"] = _safe_pearson(scores, log_client_sizes)

        scores_tensor = torch.tensor(scores, device=device, dtype=torch.float32).detach()
        fairness_values = (
            torch.tensor(client_losses, device=device, dtype=torch.float32).detach()
            if client_losses
            else None
        )
        record = {
            "layer_id": str(layer_id),
            "expert_id": str(expert_id),
            "expert_keys": expert_keys,
            "valid_client_ids": valid_client_ids,
            "features": features,
            "scores": scores_tensor,
            "fairness_values": fairness_values,
            "log_client_sizes": torch.tensor(log_client_sizes, device=device, dtype=torch.float32).detach(),
            "client_grad_states": client_grad_states,
            "ref_query": ref_query,
            "layer_id": str(layer_id),
            "expert_id": str(expert_id),
            "param_keys": param_keys,
            "metric": metric,
        }
        return metric, record

    def _build_pism_record_for_expert(
        self,
        aggregated_state,
        global_state,
        client_updates,
        client_stats,
        client_weights,
        expert_keys,
        layer_id,
        expert_id,
        global_model,
        uoc_evidences,
        reference_uoc_evidence,
        no_evidence_fallback_reason,
        device,
        named_parameters=None,
    ):
        num_classes = self._get_num_classes()
        metric = self._add_pism_metric_defaults(self._make_empty_expert_metric(None))

        if metric["score_metric"] == "delta_consensus":
            return self._build_delta_consensus_pism_record_for_expert(
                aggregated_state=aggregated_state,
                global_state=global_state,
                client_updates=client_updates,
                client_stats=client_stats,
                client_weights=client_weights,
                expert_keys=expert_keys,
                layer_id=layer_id,
                expert_id=expert_id,
                metric=metric,
                device=device,
            )

        if no_evidence_fallback_reason is not None:
            return metric, self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
                metric,
                no_evidence_fallback_reason,
            )

        query = build_stratified_query_for_expert(
            uoc_evidences,
            layer_id=layer_id,
            expert_id=expert_id,
            num_classes=num_classes,
            query_per_class=getattr(self.args, "uoc_foga_query_per_class", 4),
            min_query_samples=getattr(
                self.args,
                "uoc_foga_min_query_samples_per_expert",
                16,
            ),
            min_query_classes=getattr(self.args, "uoc_foga_min_classes_per_expert", 2),
            use_top1=True,
            seed=getattr(self.args, "seed", None),
            **self._get_query_select_kwargs(),
        )
        metric["query_size"] = int(query.get("query_size", 0))
        metric["query_num_classes"] = int(query.get("query_num_classes", 0))
        metric["query_has_residual"] = bool(query.get("has_residual", False))
        self._copy_query_stats_to_metric(metric, query)

        if query.get("fallback_reason") is not None:
            return metric, self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
                metric,
                query["fallback_reason"],
            )

        if global_model is None or not hasattr(global_model, "forward_uoc_from_hidden"):
            return metric, self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
                metric,
                "missing_global_model_forward_uoc_from_hidden",
            )

        param_keys, grad_state, grad_fallback = self._build_expert_params_and_grads(
            global_model=global_model,
            expert_keys=expert_keys,
            query=query,
            layer_id=layer_id,
            expert_id=expert_id,
            device=device,
            named_parameters=named_parameters,
        )
        if grad_fallback is not None:
            return metric, self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
                metric,
                grad_fallback,
            )

        # 诊断 A：只比较 g_query 和 reference gradient，不参与 score/PISM/聚合。
        query_ref_cos, ref_query = self._compute_query_ref_cos_diag(
            global_model=global_model,
            reference_uoc_evidence=reference_uoc_evidence,
            query_grad_state=grad_state,
            expert_keys=expert_keys,
            layer_id=layer_id,
            expert_id=expert_id,
            device=device,
            named_parameters=named_parameters,
        )
        metric["query_ref_cos"] = query_ref_cos

        score_metric = metric["score_metric"]
        expert_params = None
        if score_metric in {"grad_dot", "grad_cosine"}:
            if named_parameters is None:
                named_parameters = dict(global_model.named_parameters())
            expert_params = [named_parameters[key] for key in param_keys]

        client_scores = {}
        grad_dot_scores = []
        grad_dot_client_set_sizes = []
        cos_delta_neg_gclient_values = []
        valid_client_ids = []
        scores = []
        client_losses = []
        expert_usages = []
        total_layer_usages = []
        delta_norms = []
        expert_client_losses = []
        client_grad_sample_counts = []
        client_grad_states = []
        log_client_sizes = []
        for client_idx, client_state in enumerate(client_updates):
            delta_state = extract_expert_delta_state(
                client_state,
                global_state,
                expert_keys,
                device=device,
            )
            if score_metric in {"grad_dot", "grad_cosine"}:
                score, expert_client_loss, client_set_size, cos_delta_neg_gclient, client_grad_state = self._compute_client_grad_dot_score(
                    global_model=global_model,
                    uoc_evidences=uoc_evidences,
                    client_idx=client_idx,
                    delta_state=delta_state,
                    query_grad_state=grad_state,
                    layer_id=layer_id,
                    expert_id=expert_id,
                    expert_param_keys=param_keys,
                    expert_params=expert_params,
                    device=device,
                    score_metric=score_metric,
                )
                if score is not None:
                    grad_dot_scores.append(float(score))
                    grad_dot_client_set_sizes.append(client_set_size)
                    if cos_delta_neg_gclient is not None:
                        cos_delta_neg_gclient_values.append(float(cos_delta_neg_gclient))
            else:
                score = delta_to_negative_grad_score(
                    delta_state,
                    grad_state,
                    metric=score_metric,
                    device=device,
                )
            client_scores[client_idx] = score
            if score is None:
                continue

            valid_client_ids.append(client_idx)
            scores.append(float(score))
            client_stat = client_stats[client_idx] if client_idx < len(client_stats) else {}
            expert_usage = self._get_expert_usage_from_stats(client_stat, layer_id, expert_id)
            total_layer_usage = self._get_total_expert_usage_from_stats(client_stat, layer_id)
            if total_layer_usage < expert_usage:
                total_layer_usage = expert_usage
            client_losses.append(self._get_client_loss_from_stats(client_stat))
            expert_usages.append(expert_usage)
            total_layer_usages.append(total_layer_usage)
            delta_norms.append(l2_norm_state(delta_state, device=device))
            if score_metric == "grad_dot":
                expert_client_losses.append(expert_client_loss)
                client_grad_sample_counts.append(client_set_size)
            if score_metric == "grad_cosine":
                client_grad_states.append(client_grad_state)
            client_size = client_weights[client_idx] if client_idx < len(client_weights) else 0.0
            try:
                log_client_sizes.append(math.log1p(max(float(client_size), 0.0)))
            except (TypeError, ValueError):
                log_client_sizes.append(float("nan"))

        self._add_grad_dot_metric_stats(
            metric,
            grad_dot_scores,
            grad_dot_client_set_sizes,
            cos_delta_neg_gclient_values,
        )
        score_summary = summarize_scores(client_scores)
        metric.update(score_summary)
        valid_clients = int(score_summary["valid_score_count"])
        metric["valid_clients"] = valid_clients
        min_valid_clients = int(getattr(self.args, "uoc_foga_min_valid_clients", 2))
        if valid_clients < min_valid_clients:
            return metric, self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
                metric,
                "too_few_valid_clients",
            )

        if valid_clients < self.pism_min_clients:
            return metric, self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
                metric,
                "too_few_pism_clients",
            )

        if score_metric == "grad_dot":
            # 防御性检查：grad_cosine 配置下绝对不能误走 grad_dot 的两维 PISM 输入路径。
            if self.score_metric == "grad_cosine":
                raise ValueError(
                    "grad_cosine score_metric must not use build_grad_dot_pism_feature_tensor"
                )
            # grad_dot 的 score 由梯度内积监督；PISM 输入只使用 client 级 loss 和样本数。
            features = build_grad_dot_pism_feature_tensor(
                expert_client_loss=expert_client_losses,
                client_grad_sample_count=client_grad_sample_counts,
                device=device,
            )
            metric["pism_input_names"] = [
                "expert_client_loss",
                "log1p_client_grad_sample_count",
            ]
        elif score_metric == "grad_cosine":
            features, feature_diag = self._build_grad_cosine_pism_features_for_expert(
                client_stats=client_stats,
                valid_client_ids=valid_client_ids,
                expert_usages=expert_usages,
                total_layer_usages=total_layer_usages,
                client_grad_states=client_grad_states,
                layer_id=layer_id,
                expert_id=expert_id,
                device=device,
            )
            metric["pism_input_names"] = self._pism_input_names_for_config()
            metric.update(feature_diag)
        else:
            features = build_pism_feature_tensor(
                client_loss=client_losses,
                expert_usage=expert_usages,
                delta_norm=delta_norms,
                total_layer_usage=total_layer_usages,
                device=device,
            )
            features = self._select_pism_features_for_config(features)
            metric["pism_input_names"] = self._pism_input_names_for_config()
        if self.score_metric == "grad_cosine" and features.size(-1) != self.pism_input_dim:
            raise ValueError(
                f"grad_cosine PISM features must have input_dim={self.pism_input_dim}, "
                f"got {features.size(-1)}"
            )
        if features.size(-1) != self.pism_input_dim:
            raise ValueError(
                f"PISM feature dim mismatch: expected {self.pism_input_dim}, got {features.size(-1)}"
            )
        if self.pism_renorm_inputs and score_metric != "grad_cosine":
            features = normalize_pism_inputs(features)
        if not torch.isfinite(features).all():
            return metric, self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
                metric,
                "pism_weights_nan",
            )

        metric["pism_input_mean"] = [
            float(value) for value in features.detach().mean(dim=0).cpu().tolist()
        ]
        metric["pism_input_std"] = [
            float(value) for value in features.detach().std(dim=0, unbiased=False).cpu().tolist()
        ]
        # 诊断 B：FOGA score 是否和客户端样本量相关。
        metric["score_sample_corr"] = _safe_pearson(scores, log_client_sizes)

        scores_tensor = torch.tensor(scores, device=device, dtype=torch.float32).detach()
        if score_metric == "grad_dot":
            fairness_values = (
                torch.tensor(expert_client_losses, device=device, dtype=torch.float32).detach()
                if expert_client_losses
                else None
            )
        else:
            fairness_values = (
                torch.tensor(client_losses, device=device, dtype=torch.float32).detach()
                if client_losses
                else None
            )
        record = {
            "layer_id": str(layer_id),
            "expert_id": str(expert_id),
            "expert_keys": expert_keys,
            "valid_client_ids": valid_client_ids,
            "features": features,
            "scores": scores_tensor,
            "fairness_values": fairness_values,
            "log_client_sizes": torch.tensor(log_client_sizes, device=device, dtype=torch.float32).detach(),
            "client_grad_states": client_grad_states,
            "ref_query": ref_query,
            "layer_id": str(layer_id),
            "expert_id": str(expert_id),
            "param_keys": param_keys,
            "metric": metric,
        }
        if score_metric == "grad_dot":
            record["expert_client_losses"] = expert_client_losses
            record["client_grad_sample_counts"] = client_grad_sample_counts
        return metric, record


    def _should_collect_pism_alignment_debug(self, round_index):
        if not self.uoc_foga_pism_debug_alignment:
            return False
        if self.uoc_foga_pism_debug_alignment_max_records <= 0:
            return False
        if round_index is None:
            return True
        try:
            round_index = int(round_index)
        except (TypeError, ValueError):
            return True
        return round_index % self.uoc_foga_pism_debug_alignment_every == 0

    def _build_pism_alignment_debug_record(
        self,
        record,
        logits_tensor,
        weights_tensor,
        current_tau,
        metric,
        round_index=None,
    ):
        """构造单个 expert 的 client/feature/score/logit/weight 对齐诊断记录。"""
        client_ids = [int(client_id) for client_id in record.get("valid_client_ids", [])]
        scores = _to_flat_float_list(record.get("scores"))
        logits = _to_flat_float_list(logits_tensor)
        weights = _to_flat_float_list(weights_tensor)
        features_tensor = record.get("features")
        feature_names = list(metric.get("pism_input_names") or self._pism_input_names_for_config())
        features = []
        if torch.is_tensor(features_tensor):
            features = features_tensor.detach().cpu().float().tolist()
        else:
            try:
                features = torch.as_tensor(features_tensor, dtype=torch.float32).cpu().tolist()
            except Exception:
                features = []

        score_ranks = _rank_desc_list(scores)
        weight_ranks = _rank_desc_list(weights)
        valid_count = min(len(client_ids), len(scores), len(logits), len(weights))
        row_limit = min(valid_count, int(self.uoc_foga_pism_debug_alignment_topk))
        rows = []
        for idx in range(row_limit):
            feature_row = features[idx] if idx < len(features) and isinstance(features[idx], list) else []
            feature_map = {
                str(name): _safe_float_or_none(feature_row[pos])
                for pos, name in enumerate(feature_names)
                if pos < len(feature_row)
            }
            row = {
                "idx": int(idx),
                "client_id": int(client_ids[idx]),
                "score": _safe_float_or_none(scores[idx]),
                "score_rank_desc": _safe_rank_value_or_none(score_ranks[idx]),
                "pism_logit": _safe_float_or_none(logits[idx]),
                "pism_weight": _safe_float_or_none(weights[idx]),
                "weight_rank_desc": _safe_rank_value_or_none(weight_ranks[idx]),
                "expert_loss_z": feature_map.get("expert_loss_z"),
                "usage_ratio_z": feature_map.get("usage_ratio_z"),
                "consensus_grad_cos": feature_map.get("consensus_grad_cos"),
            }
            rows.append(row)

        foga_top_idx = None
        pism_top_idx = None
        if valid_count > 0:
            valid_scores = [
                (idx, scores[idx])
                for idx in range(valid_count)
                if _is_finite_float(scores[idx])
            ]
            valid_weights = [
                (idx, weights[idx])
                for idx in range(valid_count)
                if _is_finite_float(weights[idx])
            ]
            if valid_scores:
                foga_top_idx = max(valid_scores, key=lambda item: (item[1], -item[0]))[0]
            if valid_weights:
                pism_top_idx = max(valid_weights, key=lambda item: (item[1], -item[0]))[0]

        return {
            "round": int(round_index) if round_index is not None else None,
            "layer_id": str(record.get("layer_id")),
            "expert_id": str(record.get("expert_id")),
            "valid_clients": int(valid_count),
            "score_metric": str(self.score_metric),
            "tau": _safe_float_or_none(current_tau),
            "client_ids": client_ids[:valid_count],
            "rows": rows,
            "foga_top_client_id": (
                int(client_ids[foga_top_idx]) if foga_top_idx is not None and foga_top_idx < len(client_ids) else None
            ),
            "foga_top_score": (
                _safe_float_or_none(scores[foga_top_idx]) if foga_top_idx is not None else None
            ),
            "pism_top_client_id": (
                int(client_ids[pism_top_idx]) if pism_top_idx is not None and pism_top_idx < len(client_ids) else None
            ),
            "pism_top_score_rank": _safe_rank_value_or_none(metric.get("pism_top_client_score_rank")),
            "pism_top_score_value": _safe_float_or_none(metric.get("pism_top_client_score_value")),
            "foga_top_pism_weight": _safe_float_or_none(metric.get("foga_top_client_pism_weight")),
            "weight_score_corr": _safe_float_or_none(metric.get("weight_score_corr")),
            "logit_score_corr": _safe_float_or_none(metric.get("logit_score_corr")),
        }

    def _apply_pism_weights_for_record(
        self,
        aggregated_state,
        global_state,
        client_updates,
        record,
        meta_loss_value,
        device,
        current_tau,
        global_model=None,
        round_index=None,
    ):
        metric = record["metric"]
        with torch.no_grad():
            pism_output = self.meta_net(
                record["features"],
                tau=current_tau,
                return_logits=True,
            )
            weights_tensor = pism_output["weights"]
            logits_tensor = pism_output["logits"]
            weights_tensor = self._postprocess_pism_weights(
                weights_tensor,
                record.get("fairness_values"),
            )
        if not torch.isfinite(weights_tensor).all():
            self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                record["expert_keys"],
                metric,
                "pism_weights_nan",
            )
            return

        weight_sum = weights_tensor.sum()
        if not torch.isfinite(weight_sum) or weight_sum.item() <= 0:
            self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                record["expert_keys"],
                metric,
                "pism_weights_nan",
            )
            return

        weights_tensor = weights_tensor / weight_sum
        weights = {
            client_idx: float(weight.item())
            for client_idx, weight in zip(record["valid_client_ids"], weights_tensor)
        }
        if not self._apply_weighted_delta_for_expert(
            aggregated_state,
            global_state,
            client_updates,
            record["expert_keys"],
            weights,
            device,
        ):
            self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                record["expert_keys"],
                metric,
                "missing_client_expert_key",
            )
            return

        weight_entropy = expert_weight_entropy(weights)
        weight_min = min(weights.values()) if weights else None
        weight_max = max(weights.values()) if weights else None
        metric["weight_max"] = weight_max
        metric["weight_entropy"] = weight_entropy
        metric["aggregation_weights"] = self._client_weight_dict(weights)
        metric["aggregation_weight_source"] = "pism"
        metric["pism_used"] = True
        metric["pism_meta_loss"] = meta_loss_value
        metric["pism_weight_min"] = weight_min
        metric["pism_weight_max"] = weight_max
        metric["pism_weight_entropy"] = weight_entropy
        metric["pism_post_weight_min"] = weight_min
        metric["pism_post_weight_max"] = weight_max
        metric["pism_post_weight_entropy"] = weight_entropy
        metric["pism_fallback_reason"] = None
        metric["fallback_reason"] = None
        # 诊断 FOGA score 区分度，以及最终 PISM 权重是否和 score 对齐。
        metric.update(_build_pism_alignment_diagnostics(record["scores"], weights_tensor))
        logits_tensor = logits_tensor.detach().reshape(-1)
        metric["logit_score_corr"] = _safe_pearson(logits_tensor, record["scores"])
        metric["pism_logit_std"] = _safe_std(logits_tensor)
        if self._should_collect_pism_alignment_debug(round_index):
            metric["pism_alignment_debug_record"] = self._build_pism_alignment_debug_record(
                record=record,
                logits_tensor=logits_tensor,
                weights_tensor=weights_tensor.detach(),
                current_tau=current_tau,
                metric=metric,
                round_index=round_index,
            )
        # 诊断 B：最终 PISM weight 是否和客户端样本量相关。
        metric["weight_sample_corr"] = _safe_pearson(
            weights_tensor,
            record.get("log_client_sizes"),
        )

        # 诊断 C：FOGA top / PISM top client 的小步 expert 更新是否降低 reference loss。
        ref_query = record.get("ref_query")
        client_grad_states = record.get("client_grad_states") or []
        if (
            self.uoc_foga_ref_step_diag_enabled
            and global_model is not None
            and ref_query is not None
            and client_grad_states
        ):
            score_values = record["scores"].detach()
            if score_values.numel() == len(client_grad_states) and weights_tensor.numel() == len(client_grad_states):
                foga_top_idx = int(torch.argmax(score_values).item())
                pism_top_idx = int(torch.argmax(weights_tensor.detach()).item())
                metric["ref_step_foga_top_loss_delta"] = self._run_ref_step_diag_for_client_grad(
                    global_model=global_model,
                    ref_query=ref_query,
                    layer_id=record.get("layer_id"),
                    expert_id=record.get("expert_id"),
                    expert_param_keys=record.get("param_keys"),
                    client_grad_state=client_grad_states[foga_top_idx],
                    device=device,
                )
                metric["ref_step_pism_top_loss_delta"] = self._run_ref_step_diag_for_client_grad(
                    global_model=global_model,
                    ref_query=ref_query,
                    layer_id=record.get("layer_id"),
                    expert_id=record.get("expert_id"),
                    expert_param_keys=record.get("param_keys"),
                    client_grad_state=client_grad_states[pism_top_idx],
                    device=device,
                )

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        if len(client_updates) == 0:
            raise ValueError("UOCFOGAPISMExpertAlignAggregator requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")

        round_index = kwargs.get("round_index", None)
        current_tau = self._get_current_pism_tau(round_index=round_index)

        self._build_key_cache(client_updates[0].keys())
        global_state = self._get_global_state(client_updates, global_model)
        aggregated_state = {}
        self._aggregate_non_expert_keys(aggregated_state, client_updates, client_weights)

        uoc_evidences = self._resolve_uoc_evidences(kwargs)
        reference_uoc_evidence = kwargs.get("reference_uoc_evidence", None)
        no_evidence_fallback_reason = None
        if not uoc_evidences:
            no_evidence_fallback_reason = "no_uoc_evidence_passed_to_aggregator"

        client_stats = self._normalize_client_stats(kwargs.get("client_stats"), len(client_updates))
        device = self._get_model_device(global_model)
        # 每轮只构造一次参数字典，供所有 expert 的 query 梯度复用。
        named_parameters = dict(global_model.named_parameters()) if global_model is not None else None
        self.meta_net.to(device)
        self._move_meta_optimizer_state_to_device(device)

        uoc_foga_stats = {}
        per_expert_records = []
        for (layer_id, expert_id), expert_keys in sorted(
            self._expert_key_cache.items(),
            key=lambda item: (int(item[0][0]), int(item[0][1])),
        ):
            layer_stats = uoc_foga_stats.setdefault(str(layer_id), {})
            metric, record = self._build_pism_record_for_expert(
                aggregated_state=aggregated_state,
                global_state=global_state,
                client_updates=client_updates,
                client_stats=client_stats,
                client_weights=client_weights,
                expert_keys=expert_keys,
                layer_id=str(layer_id),
                expert_id=str(expert_id),
                global_model=global_model,
                uoc_evidences=uoc_evidences,
                reference_uoc_evidence=reference_uoc_evidence,
                no_evidence_fallback_reason=no_evidence_fallback_reason,
                device=device,
                named_parameters=named_parameters,
            )
            layer_stats[str(expert_id)] = metric
            if record is not None:
                per_expert_records.append(record)

        meta_loss_value = None
        meta_loss_failed = False
        successful_meta_losses = []
        successful_meta_steps = 0
        if per_expert_records:
            for _ in range(self.pism_meta_steps):
                meta_losses = []
                for record in per_expert_records:
                    weights = self.meta_net(record["features"], tau=current_tau)
                    meta_losses.append(-(weights * record["scores"]).mean())
                meta_loss = torch.stack(meta_losses).mean()

                if not torch.isfinite(meta_loss):
                    if successful_meta_steps == 0:
                        meta_loss_failed = True
                    break

                self.meta_optimizer.zero_grad()
                meta_loss.backward()
                self.meta_optimizer.step()
                self.pism_update_steps += 1
                successful_meta_steps += 1
                successful_meta_losses.append(float(meta_loss.detach().cpu().item()))

            if successful_meta_steps > 0:
                meta_loss_value = sum(successful_meta_losses) / len(successful_meta_losses)
            else:
                meta_loss_value = None
        else:
            meta_loss_failed = True

        if meta_loss_failed:
            for record in per_expert_records:
                self._fallback_expert(
                    aggregated_state,
                    global_state,
                    client_updates,
                    record["expert_keys"],
                    record["metric"],
                    "pism_meta_loss_nan" if per_expert_records else "pism_no_valid_records",
                )
        else:
            for record in per_expert_records:
                self._apply_pism_weights_for_record(
                    aggregated_state,
                    global_state,
                    client_updates,
                    record,
                    meta_loss_value,
                    device,
                    current_tau,
                    global_model=global_model,
                    round_index=round_index,
                )

        # 只汇总轻量 Python 标量，便于 server 日志观察 PISM 趋势。
        expert_metrics = [
            expert_metric
            for layer_stats in uoc_foga_stats.values()
            for expert_metric in layer_stats.values()
        ]
        total_experts = len(expert_metrics)
        pism_meta_losses = []
        pism_weight_entropies = []
        pism_weight_max_values = []
        pism_post_weight_min_values = []
        pism_post_weight_max_values = []
        pism_post_weight_entropy_values = []
        pism_score_stds = []
        pism_score_pos_fracs = []
        pism_weight_score_corrs = []
        pism_top_score_ranks = []
        pism_top_score_values = []
        foga_top_pism_weights = []
        pism_consensus_grad_cos_means = []
        pism_consensus_grad_pos_fracs = []
        pism_expert_loss_z_stds = []
        query_ref_cos_values = []
        pism_score_sample_corrs = []
        pism_weight_sample_corrs = []
        pism_logit_score_corrs = []
        pism_logit_score_corr_valid_count = 0
        pism_logit_stds = []
        pism_alignment_debug_candidates = []
        ref_step_foga_top_loss_deltas = []
        ref_step_pism_top_loss_deltas = []
        # mixed_global_expert query 的 round 级诊断。
        # 这些统计来自每个 expert 的 D_query,l,e 构造结果，用来确认
        # global-balanced 与 expert-specific 两部分是否真的混入成功。
        mixed_global_query_sizes = []
        mixed_expert_query_sizes = []
        mixed_global_query_num_classes = []
        mixed_expert_query_num_classes = []
        mixed_global_ratio_effective_values = []
        pism_weight_score_corr_valid_count = 0
        pism_fallback_reason_counts = collections.defaultdict(int)
        updated_experts = 0

        for expert_metric in expert_metrics:
            pism_used = bool(expert_metric.get("pism_used", False))
            fallback_reason = expert_metric.get("fallback_reason")
            pism_fallback_reason = expert_metric.get("pism_fallback_reason")
            reason = (
                pism_fallback_reason
                if pism_fallback_reason is not None
                else fallback_reason
                if fallback_reason is not None
                else "none"
            )
            pism_fallback_reason_counts[str(reason)] += 1

            # mixed query 统计不依赖 PISM 是否最终成功更新；只要当前 expert
            # 构造过 D_query,l,e，就纳入 round summary，方便诊断 query set 本身。
            mixed_global_query_sizes.append(expert_metric.get("mixed_global_query_size"))
            mixed_expert_query_sizes.append(expert_metric.get("mixed_expert_query_size"))
            mixed_global_query_num_classes.append(expert_metric.get("mixed_global_query_num_classes"))
            mixed_expert_query_num_classes.append(expert_metric.get("mixed_expert_query_num_classes"))
            mixed_global_ratio_effective_values.append(expert_metric.get("mixed_global_ratio_effective"))

            if not pism_used or fallback_reason is not None:
                continue

            updated_experts += 1
            meta_loss_value = expert_metric.get("pism_meta_loss")
            if meta_loss_value is not None:
                pism_meta_losses.append(float(meta_loss_value))
            entropy_value = expert_metric.get("pism_weight_entropy")
            if entropy_value is not None:
                pism_weight_entropies.append(float(entropy_value))
            weight_max_value = expert_metric.get("pism_weight_max")
            if weight_max_value is not None:
                pism_weight_max_values.append(float(weight_max_value))
            post_weight_min_value = expert_metric.get("pism_post_weight_min")
            if post_weight_min_value is not None:
                pism_post_weight_min_values.append(float(post_weight_min_value))
            post_weight_max_value = expert_metric.get("pism_post_weight_max")
            if post_weight_max_value is not None:
                pism_post_weight_max_values.append(float(post_weight_max_value))
            post_weight_entropy_value = expert_metric.get("pism_post_weight_entropy")
            if post_weight_entropy_value is not None:
                pism_post_weight_entropy_values.append(float(post_weight_entropy_value))
            pism_score_stds.append(expert_metric.get("score_std"))
            pism_score_pos_fracs.append(expert_metric.get("score_pos_frac"))
            weight_score_corr = expert_metric.get("weight_score_corr")
            pism_weight_score_corrs.append(weight_score_corr)
            if _safe_numeric_values([weight_score_corr]):
                pism_weight_score_corr_valid_count += 1
            pism_top_score_ranks.append(expert_metric.get("pism_top_client_score_rank"))
            pism_top_score_values.append(expert_metric.get("pism_top_client_score_value"))
            foga_top_pism_weights.append(expert_metric.get("foga_top_client_pism_weight"))
            pism_consensus_grad_cos_means.append(expert_metric.get("consensus_grad_cos_mean"))
            pism_consensus_grad_pos_fracs.append(expert_metric.get("consensus_grad_pos_frac"))
            pism_expert_loss_z_stds.append(expert_metric.get("expert_loss_z_std"))
            query_ref_cos_values.append(expert_metric.get("query_ref_cos"))
            pism_score_sample_corrs.append(expert_metric.get("score_sample_corr"))
            pism_weight_sample_corrs.append(expert_metric.get("weight_sample_corr"))
            logit_score_corr = expert_metric.get("logit_score_corr")
            pism_logit_score_corrs.append(logit_score_corr)
            if _safe_numeric_values([logit_score_corr]):
                pism_logit_score_corr_valid_count += 1
            pism_logit_stds.append(expert_metric.get("pism_logit_std"))
            debug_record = expert_metric.get("pism_alignment_debug_record")
            if isinstance(debug_record, dict):
                pism_alignment_debug_candidates.append(debug_record)
            ref_step_foga_top_loss_deltas.append(expert_metric.get("ref_step_foga_top_loss_delta"))
            ref_step_pism_top_loss_deltas.append(expert_metric.get("ref_step_pism_top_loss_delta"))

        fallback_experts = total_experts - updated_experts
        pism_alignment_debug_records = []
        if self.uoc_foga_pism_debug_alignment and self.uoc_foga_pism_debug_alignment_max_records > 0:
            def _debug_priority(record):
                rank = record.get("pism_top_score_rank")
                try:
                    rank_value = float(rank)
                except (TypeError, ValueError):
                    rank_value = float("nan")
                rank_bad = math.isfinite(rank_value) and rank_value > 1.0
                # 先打印 PISM top 没有对齐 FOGA top 的 record；rank 越差越靠前。
                return (0 if rank_bad else 1, -rank_value if math.isfinite(rank_value) else 0.0)
            pism_alignment_debug_records = sorted(
                pism_alignment_debug_candidates,
                key=_debug_priority,
            )[: self.uoc_foga_pism_debug_alignment_max_records]

        pism_summary = {
            "uoc_foga_score_metric": self.score_metric,
            "uoc_foga_pism_input_dim": int(self.pism_input_dim),
            "uoc_foga_pism_meta_loss_mean": (
                sum(pism_meta_losses) / len(pism_meta_losses)
                if pism_meta_losses
                else None
            ),
            "uoc_foga_pism_updated_experts": updated_experts,
            "uoc_foga_pism_fallback_experts": fallback_experts,
            "uoc_foga_pism_fallback_reason_counts": dict(pism_fallback_reason_counts),
            "uoc_foga_pism_weight_entropy_mean": (
                sum(pism_weight_entropies) / len(pism_weight_entropies)
                if pism_weight_entropies
                else None
            ),
            "uoc_foga_pism_weight_max_mean": (
                sum(pism_weight_max_values) / len(pism_weight_max_values)
                if pism_weight_max_values
                else None
            ),
            "uoc_foga_pism_min_weight_factor": float(
                self.uoc_foga_pism_min_weight_factor
            ),
            "uoc_foga_pism_fairness_blend": float(
                self.uoc_foga_pism_fairness_blend
            ),
            "pism_post_weight_min_mean": (
                sum(pism_post_weight_min_values) / len(pism_post_weight_min_values)
                if pism_post_weight_min_values
                else None
            ),
            "pism_post_weight_max_mean": (
                sum(pism_post_weight_max_values) / len(pism_post_weight_max_values)
                if pism_post_weight_max_values
                else None
            ),
            "pism_post_weight_entropy_mean": (
                sum(pism_post_weight_entropy_values) / len(pism_post_weight_entropy_values)
                if pism_post_weight_entropy_values
                else None
            ),
            "uoc_foga_pism_score_std_mean": _safe_mean(pism_score_stds),
            "uoc_foga_pism_score_pos_frac_mean": _safe_mean(pism_score_pos_fracs),
            "uoc_foga_pism_weight_score_corr_mean": _safe_mean(pism_weight_score_corrs),
            "uoc_foga_pism_weight_score_corr_valid_frac": (
                pism_weight_score_corr_valid_count / updated_experts
                if updated_experts > 0
                else 0.0
            ),
            "uoc_foga_pism_logit_score_corr_mean": _safe_mean(pism_logit_score_corrs),
            "uoc_foga_pism_logit_score_corr_valid_frac": (
                pism_logit_score_corr_valid_count / updated_experts
                if updated_experts > 0
                else 0.0
            ),
            "uoc_foga_pism_logit_std_mean": _safe_mean(pism_logit_stds),
            "pism_alignment_debug_records": pism_alignment_debug_records,
            "uoc_foga_pism_pism_top_score_rank_mean": _safe_mean(pism_top_score_ranks),
            "uoc_foga_pism_pism_top_score_value_mean": _safe_mean(pism_top_score_values),
            "uoc_foga_pism_foga_top_pism_weight_mean": _safe_mean(foga_top_pism_weights),
            "uoc_foga_pism_consensus_grad_cos_mean": _safe_mean(pism_consensus_grad_cos_means),
            "uoc_foga_pism_consensus_grad_pos_frac_mean": _safe_mean(pism_consensus_grad_pos_fracs),
            "uoc_foga_pism_expert_loss_z_std_mean": _safe_mean(pism_expert_loss_z_stds),
            "uoc_foga_query_ref_cos_mean": _safe_mean(query_ref_cos_values),
            "uoc_foga_query_ref_cos_std": _safe_std(query_ref_cos_values),
            "uoc_foga_query_ref_cos_min": min(_safe_numeric_values(query_ref_cos_values)) if _safe_numeric_values(query_ref_cos_values) else float("nan"),
            "uoc_foga_query_ref_cos_valid_frac": (len(_safe_numeric_values(query_ref_cos_values)) / total_experts if total_experts > 0 else 0.0),
            "uoc_foga_pism_score_sample_corr_mean": _safe_mean(pism_score_sample_corrs),
            "uoc_foga_pism_score_sample_corr_valid_frac": (len(_safe_numeric_values(pism_score_sample_corrs)) / updated_experts if updated_experts > 0 else 0.0),
            "uoc_foga_pism_weight_sample_corr_mean": _safe_mean(pism_weight_sample_corrs),
            "uoc_foga_pism_weight_sample_corr_valid_frac": (len(_safe_numeric_values(pism_weight_sample_corrs)) / updated_experts if updated_experts > 0 else 0.0),
            "uoc_foga_ref_step_foga_top_loss_delta_mean": _safe_mean(ref_step_foga_top_loss_deltas),
            "uoc_foga_ref_step_foga_top_improve_frac": (sum(1 for value in _safe_numeric_values(ref_step_foga_top_loss_deltas) if value < 0.0) / len(_safe_numeric_values(ref_step_foga_top_loss_deltas)) if _safe_numeric_values(ref_step_foga_top_loss_deltas) else float("nan")),
            "uoc_foga_ref_step_pism_top_loss_delta_mean": _safe_mean(ref_step_pism_top_loss_deltas),
            "uoc_foga_ref_step_pism_top_improve_frac": (sum(1 for value in _safe_numeric_values(ref_step_pism_top_loss_deltas) if value < 0.0) / len(_safe_numeric_values(ref_step_pism_top_loss_deltas)) if _safe_numeric_values(ref_step_pism_top_loss_deltas) else float("nan")),
            "uoc_foga_ref_step_valid_frac": (max(len(_safe_numeric_values(ref_step_foga_top_loss_deltas)), len(_safe_numeric_values(ref_step_pism_top_loss_deltas))) / updated_experts if updated_experts > 0 else 0.0),
            "uoc_foga_mixed_global_query_size_mean": _safe_mean(mixed_global_query_sizes),
            "uoc_foga_mixed_expert_query_size_mean": _safe_mean(mixed_expert_query_sizes),
            "uoc_foga_mixed_global_query_num_classes_mean": _safe_mean(mixed_global_query_num_classes),
            "uoc_foga_mixed_expert_query_num_classes_mean": _safe_mean(mixed_expert_query_num_classes),
            "uoc_foga_mixed_global_ratio_effective_mean": _safe_mean(mixed_global_ratio_effective_values),
            "uoc_foga_pism_used_frac": (
                updated_experts / total_experts
                if total_experts > 0
                else 0.0
            ),
            "uoc_foga_pism_update_steps": int(self.pism_update_steps),
            "uoc_foga_pism_meta_steps": int(self.pism_meta_steps),
            "uoc_foga_pism_meta_steps_successful": int(successful_meta_steps),
            "uoc_foga_pism_meta_steps_requested": int(self.pism_meta_steps),
            "uoc_foga_pism_tau_schedule": self.pism_tau_schedule,
            "uoc_foga_pism_tau": float(current_tau),
            "uoc_foga_pism_tau_init": float(self.pism_tau_init),
            "uoc_foga_pism_tau_min": float(self.pism_tau_min),
            "uoc_foga_pism_tau_decay": float(self.pism_tau_decay),
            "uoc_foga_client_grad_query_per_class": int(
                self.uoc_foga_client_grad_query_per_class
            ),
            "uoc_foga_client_grad_min_samples_per_expert": int(
                self.uoc_foga_client_grad_min_samples_per_expert
            ),
            "uoc_foga_client_grad_min_classes_per_expert": int(
                self.uoc_foga_client_grad_min_classes_per_expert
            ),
            "uoc_foga_client_grad_min_expert_token_ratio": float(
                self.uoc_foga_client_grad_min_expert_token_ratio
            ),
            "uoc_foga_client_grad_max_samples_per_client_per_class": int(
                self.uoc_foga_client_grad_max_samples_per_client_per_class
            ),
            "uoc_foga_client_grad_fallback_to_random": bool(
                self.uoc_foga_client_grad_fallback_to_random
            ),
        }
        self.last_aggregation_metrics = {
            "uoc_foga_stats": uoc_foga_stats,
            "uoc_foga_pism_summary": pism_summary,
            "expert_aggregation_weights": self._build_expert_aggregation_weights_summary(
                uoc_foga_stats,
                weight_source="pism",
            ),
            "uoc_foga_pism_meta_loss_mean": pism_summary["uoc_foga_pism_meta_loss_mean"],
            "uoc_foga_pism_updated_experts": updated_experts,
            "uoc_foga_pism_fallback_experts": fallback_experts,
        }
        return collections.OrderedDict(
            (key, aggregated_state[key])
            for key in client_updates[0].keys()
        )


def build_aggregator(args):
    expert_method = getattr(args, "expert_agg_method", "sample_weighted")
    if expert_method == "uoc_foga_expert_align":
        return UOCFOGAExpertAlignAggregator(
            args=args,
            non_expert_method=args.non_expert_agg_method,
        )
    if expert_method == "uoc_foga_pism_expert_align":
        return UOCFOGAPISMExpertAlignAggregator(args=args)

    return SplitParameterAggregator(
        non_expert_method=args.non_expert_agg_method,
        expert_method=expert_method,
    )
