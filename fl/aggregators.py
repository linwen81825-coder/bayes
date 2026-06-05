import collections
from abc import ABC, abstractmethod

import torch
from torch import nn

from fl.pism import ExpertPISM, build_pism_feature_tensor, normalize_pism_inputs
from fl.uoc_foga import (
    build_stratified_query_for_expert,
    cosine_delta_to_negative_grad,
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

        aggregated_state = collections.OrderedDict()
        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()
            if not torch.is_floating_point(first_value):
                # 非浮点 buffer 通常不能加权平均，沿用第一个客户端的值。
                aggregated_state[key] = first_value.clone()
                continue

            if is_expert_parameter(key):
                normalized_weights = expert_weights
            else:
                normalized_weights = non_expert_weights

            aggregated_state[key] = torch.zeros_like(first_value)
            for update, weight in zip(client_updates, normalized_weights):
                aggregated_state[key] += update[key].detach().cpu() * weight

        self.last_aggregation_metrics = {}
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
            "weight_max": None,
            "weight_entropy": None,
            "fallback_reason": fallback_reason,
        }

    def _build_expert_params_and_grads(
        self,
        global_model,
        expert_keys,
        query,
        layer_id,
        expert_id,
        device,
    ):
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
        loss = nn.CrossEntropyLoss()(output["logits"], labels)
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
    ):
        num_classes = self._get_num_classes()
        metric = self._make_empty_expert_metric(None)

        if no_evidence_fallback_reason is not None:
            metric["fallback_reason"] = no_evidence_fallback_reason
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
        )
        metric["query_size"] = int(query.get("query_size", 0))
        metric["query_num_classes"] = int(query.get("query_num_classes", 0))
        metric["query_has_residual"] = bool(query.get("has_residual", False))

        if query.get("fallback_reason") is not None:
            metric["fallback_reason"] = query["fallback_reason"]
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        if global_model is None or not hasattr(global_model, "forward_uoc_from_hidden"):
            metric["fallback_reason"] = "missing_global_model_forward_uoc_from_hidden"
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
        )
        if grad_fallback is not None:
            metric["fallback_reason"] = grad_fallback
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        client_scores = {}
        for client_idx, client_state in enumerate(client_updates):
            delta_state = extract_expert_delta_state(
                client_state,
                global_state,
                expert_keys,
                device=device,
            )
            score = cosine_delta_to_negative_grad(
                delta_state,
                grad_state,
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
            self._apply_uniform_delta_for_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
            )
            return metric

        metric["weight_max"] = max(weights.values()) if weights else None
        metric["weight_entropy"] = expert_weight_entropy(weights)
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
            )
            layer_stats[str(expert_id)] = metric

        self.last_aggregation_metrics = {"uoc_foga_stats": uoc_foga_stats}
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
        self.meta_net = ExpertPISM(
            input_dim=getattr(args, "uoc_foga_pism_input_dim", 3),
            hidden_size=getattr(args, "uoc_foga_pism_hidden_size", 64),
            dropout=getattr(args, "uoc_foga_pism_dropout", 0.0),
        )
        self.meta_optimizer = torch.optim.Adam(
            self.meta_net.parameters(),
            lr=getattr(args, "uoc_foga_pism_lr", 1e-3),
        )

    def _add_pism_metric_defaults(self, metric):
        metric.update({
            "pism_used": False,
            "pism_meta_loss": None,
            "pism_weight_max": None,
            "pism_weight_entropy": None,
            "pism_input_mean": None,
            "pism_input_std": None,
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
        )
        metric["query_size"] = int(query.get("query_size", 0))
        metric["query_num_classes"] = int(query.get("query_num_classes", 0))
        metric["query_has_residual"] = bool(query.get("has_residual", False))

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

        client_scores = {}
        valid_client_ids = []
        scores = []
        client_losses = []
        expert_usages = []
        delta_norms = []
        for client_idx, client_state in enumerate(client_updates):
            delta_state = extract_expert_delta_state(
                client_state,
                global_state,
                expert_keys,
                device=device,
            )
            score = cosine_delta_to_negative_grad(
                delta_state,
                grad_state,
                device=device,
            )
            client_scores[client_idx] = score
            if score is None:
                continue

            valid_client_ids.append(client_idx)
            scores.append(float(score))
            client_stat = client_stats[client_idx] if client_idx < len(client_stats) else {}
            client_losses.append(self._get_client_loss_from_stats(client_stat))
            expert_usages.append(
                self._get_expert_usage_from_stats(client_stat, layer_id, expert_id)
            )
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

        pism_min_clients = int(getattr(self.args, "uoc_foga_pism_min_clients", 2))
        if valid_clients < pism_min_clients:
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
            device=device,
        )
        if bool(getattr(self.args, "uoc_foga_pism_renorm_inputs", True)):
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
    ):
        metric = record["metric"]
        tau = float(getattr(self.args, "uoc_foga_pism_tau", 1.0))
        with torch.no_grad():
            weights_tensor = self.meta_net(record["features"], tau=tau)
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
        self.meta_net.to(device)

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
            )
            layer_stats[str(expert_id)] = metric
            if record is not None:
                per_expert_records.append(record)

        meta_loss_value = None
        meta_loss_failed = False
        if per_expert_records:
            tau = float(getattr(self.args, "uoc_foga_pism_tau", 1.0))
            meta_losses = []
            for record in per_expert_records:
                weights = self.meta_net(record["features"], tau=tau)
                meta_losses.append(-(weights * record["scores"]).sum())
            meta_loss = torch.stack(meta_losses).mean()
            if not torch.isfinite(meta_loss):
                meta_loss_failed = True
                meta_loss_value = None
            else:
                self.meta_optimizer.zero_grad()
                meta_loss.backward()
                self.meta_optimizer.step()
                meta_loss_value = float(meta_loss.detach().cpu().item())
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
                )

        pism_meta_losses = [
            expert_metric.get("pism_meta_loss")
            for layer_stats in uoc_foga_stats.values()
            for expert_metric in layer_stats.values()
            if expert_metric.get("pism_meta_loss") is not None
        ]
        updated_experts = sum(
            1
            for layer_stats in uoc_foga_stats.values()
            for expert_metric in layer_stats.values()
            if expert_metric.get("pism_used")
        )
        fallback_experts = sum(
            1
            for layer_stats in uoc_foga_stats.values()
            for expert_metric in layer_stats.values()
            if expert_metric.get("fallback_reason") is not None
        )
        self.last_aggregation_metrics = {
            "uoc_foga_stats": uoc_foga_stats,
            "uoc_foga_pism_meta_loss_mean": (
                sum(pism_meta_losses) / len(pism_meta_losses)
                if pism_meta_losses
                else None
            ),
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
