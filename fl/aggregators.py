import collections
from abc import ABC, abstractmethod

import torch
from torch import nn

from fl.pism import ExpertPISM, build_pism_feature_tensor, normalize_pism_inputs
from fl.uoc_foga import (
    build_stratified_query_for_expert,
    delta_to_negative_grad_score,
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
        if score_metric not in {"cosine", "dot"}:
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
        ):
            if key in query:
                metric[key] = query[key]

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
        no_evidence_fallback_reason,
        device,
        named_parameters=None,
    ):
        num_classes = self._get_num_classes()
        metric = self._make_empty_expert_metric(None)

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

        _, grad_state, grad_fallback = self._build_expert_params_and_grads(
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
        client_scores = {}
        for client_idx, client_state in enumerate(client_updates):
            delta_state = extract_expert_delta_state(
                client_state,
                global_state,
                expert_keys,
                device=device,
            )
            score = delta_to_negative_grad_score(
                delta_state,
                grad_state,
                metric=score_metric,
                device=device,
            )
            client_scores[client_idx] = score

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


class UOCFOGAPISMExpertAlignAggregator(UOCFOGAExpertAlignAggregator):
    # PISM 版 UOC-FOGA：用 DeepSets 元网络从 client/expert 特征生成专家聚合权重。
    def __init__(self, args):
        super(UOCFOGAPISMExpertAlignAggregator, self).__init__(
            args=args,
            non_expert_method=args.non_expert_agg_method,
        )
        self.pism_input_dim = int(getattr(args, "uoc_foga_pism_input_dim", 5))
        if self.pism_input_dim != 5:
            raise ValueError(
                "当前版本 PISM features 是 5 维，需要设置 uoc_foga_pism_input_dim: 5."
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
            "pism_input_names": [
                "client_loss",
                "log1p_expert_usage",
                "expert_usage_ratio",
                "log1p_delta_norm",
                "log1p_delta_norm_per_sqrt_usage",
            ],
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

    def _build_pism_record_for_expert(
        self,
        aggregated_state,
        global_state,
        client_updates,
        client_stats,
        expert_keys,
        layer_id,
        expert_id,
        global_model,
        uoc_evidences,
        no_evidence_fallback_reason,
        device,
        named_parameters=None,
    ):
        num_classes = self._get_num_classes()
        metric = self._add_pism_metric_defaults(self._make_empty_expert_metric(None))

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

        _, grad_state, grad_fallback = self._build_expert_params_and_grads(
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

        score_metric = metric["score_metric"]
        client_scores = {}
        valid_client_ids = []
        scores = []
        client_losses = []
        expert_usages = []
        total_layer_usages = []
        delta_norms = []
        for client_idx, client_state in enumerate(client_updates):
            delta_state = extract_expert_delta_state(
                client_state,
                global_state,
                expert_keys,
                device=device,
            )
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

        features = build_pism_feature_tensor(
            client_loss=client_losses,
            expert_usage=expert_usages,
            delta_norm=delta_norms,
            total_layer_usage=total_layer_usages,
            device=device,
        )
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
        scores_tensor = torch.tensor(scores, device=device, dtype=torch.float32).detach()
        record = {
            "layer_id": str(layer_id),
            "expert_id": str(expert_id),
            "expert_keys": expert_keys,
            "valid_client_ids": valid_client_ids,
            "features": features,
            "scores": scores_tensor,
            "metric": metric,
        }
        return metric, record

    def _apply_pism_weights_for_record(
        self,
        aggregated_state,
        global_state,
        client_updates,
        record,
        meta_loss_value,
        device,
        current_tau,
    ):
        metric = record["metric"]
        with torch.no_grad():
            weights_tensor = self.meta_net(record["features"], tau=current_tau)
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
        weight_max = max(weights.values()) if weights else None
        metric["weight_max"] = weight_max
        metric["weight_entropy"] = weight_entropy
        metric["aggregation_weights"] = self._client_weight_dict(weights)
        metric["aggregation_weight_source"] = "pism"
        metric["pism_used"] = True
        metric["pism_meta_loss"] = meta_loss_value
        metric["pism_weight_max"] = weight_max
        metric["pism_weight_entropy"] = weight_entropy
        metric["pism_fallback_reason"] = None
        metric["fallback_reason"] = None

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
                expert_keys=expert_keys,
                layer_id=str(layer_id),
                expert_id=str(expert_id),
                global_model=global_model,
                uoc_evidences=uoc_evidences,
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
                    meta_losses.append(-(weights * record["scores"]).sum())
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

        fallback_experts = total_experts - updated_experts
        pism_summary = {
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
