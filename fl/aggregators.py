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
            "fallback_reason": fallback_reason,
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
            named_parameters=named_parameters,
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
        self.pism_input_dim = int(getattr(args, "uoc_foga_pism_input_dim", 3))
        self.pism_hidden_size = int(getattr(args, "uoc_foga_pism_hidden_size", 64))
        self.pism_dropout = float(getattr(args, "uoc_foga_pism_dropout", 0.0))
        self.pism_lr = float(getattr(args, "uoc_foga_pism_lr", 1e-3))
        self.pism_tau = float(getattr(args, "uoc_foga_pism_tau", 1.0))
        self.pism_renorm_inputs = bool(getattr(args, "uoc_foga_pism_renorm_inputs", True))
        self.pism_min_clients = int(getattr(args, "uoc_foga_pism_min_clients", 2))
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
            "pism_fallback_reason": None,
            # 下面都是纯诊断字段，不参与聚合权重计算。
            "pism_input_names": None,
            "pism_raw_input_mean": None,
            "pism_raw_input_std": None,
            "pism_raw_input_min": None,
            "pism_raw_input_max": None,
            "pism_raw_input_p10": None,
            "pism_raw_input_p50": None,
            "pism_raw_input_p90": None,
            "pism_raw_input_zero_frac": None,
            "pism_raw_input_nonfinite_frac": None,
            "pism_raw_input_corr_matrix": None,
            "pism_raw_input_score_corr": None,
            "pism_norm_input_mean": None,
            "pism_norm_input_std": None,
            "pism_norm_input_min": None,
            "pism_norm_input_max": None,
            "pism_norm_input_p10": None,
            "pism_norm_input_p50": None,
            "pism_norm_input_p90": None,
            "pism_norm_input_zero_frac": None,
            "pism_norm_input_nonfinite_frac": None,
            "pism_norm_input_corr_matrix": None,
            "pism_norm_input_score_corr": None,
            "pism_feature_collapse_frac": None,
            "pism_logits_mean": None,
            "pism_logits_std": None,
            "pism_logits_min": None,
            "pism_logits_max": None,
            "pism_weight_score_corr": None,
            "pism_weight_input_corr": None,
            "pism_logit_input_corr": None,
            "pism_top_score_client_id": None,
            "pism_top_weight_client_id": None,
            "pism_top_weight_matches_top_score": None,
            "pism_feature_sensitivity_l1": None,
            "pism_feature_sensitivity_kl": None,
            "pism_feature_sensitivity_top_change": None,
            "pism_feature_grad_abs_mean": None,
            "pism_first_layer_weight_norm_by_input": None,
            "pism_first_layer_grad_norm_by_input": None,
            "pism_diag_alignment_loss_before_step": None,
            "pism_diag_alignment_loss_after_step": None,
            "pism_diag_alignment_loss_delta": None,
            "pism_score_sample_corr": None,
            "pism_weight_sample_corr": None,
            "pism_logit_sample_corr": None,
            "pism_delta_norm_sample_corr": None,
            "pism_usage_sample_corr": None,
            "pism_weight_usage_corr": None,
            "pism_weight_delta_norm_corr": None,
            "pism_weight_loss_corr": None,
            "pism_logit_usage_corr": None,
            "pism_logit_delta_norm_corr": None,
            "pism_logit_loss_corr": None,
            "pism_agg_delta_query_cos": None,
            "foga_agg_delta_query_cos": None,
            "uniform_agg_delta_query_cos": None,
            "pism_vs_uniform_agg_delta_query_cos_gap": None,
            "pism_vs_foga_agg_delta_query_cos_gap": None,
            "pism_weight_top1": None,
            "pism_weight_top2_sum": None,
            "pism_weight_eff_clients": None,
            "pism_weight_gini": None,
            "pism_top_minus_foga_top_score": None,
            "pism_top_minus_foga_top_usage": None,
            "pism_top_minus_foga_top_delta_norm": None,
            "pism_top_minus_foga_top_sample_size": None,
            "pism_foga_weight_l1": None,
            "pism_foga_weight_kl": None,
            "pism_foga_top_match": None,
            "pism_foga_rank_corr": None,
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

    def _pism_input_names_for_config(self):
        """返回当前 PISM 配置实际喂入的特征名。"""
        score_metric = str(getattr(self.args, "uoc_foga_score_metric", "cosine"))
        if score_metric in {"cosine", "delta_consensus"} and self.pism_input_dim == 3:
            return [
                "client_loss",
                "log1p_expert_usage_ratio",
                "log1p_delta_norm",
            ]

        base_names = [
            "client_loss",
            "log1p_expert_usage",
            "log1p_delta_norm",
        ]
        if self.pism_input_dim <= len(base_names):
            return base_names[: self.pism_input_dim]
        return base_names + [
            f"extra_feature_{idx}"
            for idx in range(len(base_names), self.pism_input_dim)
        ]

    def _pism_diag_input_names(self):
        """返回当前 PISM 实际使用的输入名。"""
        return self._pism_input_names_for_config()

    def _select_pism_features_for_config(self, features):
        """只在 old1 三维 PISM 输入下把 usage 替换成 usage ratio 的 log1p。"""
        score_metric = str(getattr(self.args, "uoc_foga_score_metric", "cosine"))
        if score_metric in {"cosine", "delta_consensus"} and self.pism_input_dim == 3:
            client_loss = features[..., 0]
            expert_usage_ratio = features[..., 2]
            log1p_expert_usage_ratio = torch.log1p(expert_usage_ratio.clamp_min(0.0))
            log1p_delta_norm = features[..., 3]
            return torch.stack(
                [client_loss, log1p_expert_usage_ratio, log1p_delta_norm],
                dim=-1,
            )
        return features

    def _pism_diag_safe_float(self, value):
        """把 tensor / number 安全转成 Python float，失败时返回 None。"""
        if value is None:
            return None
        try:
            if torch.is_tensor(value):
                if value.numel() != 1:
                    return None
                value = value.detach().float().cpu().item()
            value = float(value)
        except (TypeError, ValueError, RuntimeError):
            return None
        if value != value:
            return None
        return value

    def _pism_diag_safe_pearson_corr(self, x, y, eps=1e-12):
        """安全计算 Pearson 相关系数。"""
        if x is None or y is None:
            return None

        x = torch.as_tensor(x).detach().float().reshape(-1)
        y = torch.as_tensor(y).detach().float().reshape(-1)
        if x.numel() != y.numel() or x.numel() < 2:
            return None

        finite_mask = torch.isfinite(x) & torch.isfinite(y)
        if int(finite_mask.sum().item()) < 2:
            return None

        x = x[finite_mask]
        y = y[finite_mask]
        x = x - x.mean()
        y = y - y.mean()
        denom = x.norm() * y.norm()
        if (not torch.isfinite(denom)) or denom.item() <= eps:
            return None
        corr = (x * y).sum() / denom
        corr = corr.clamp(min=-1.0, max=1.0)
        return float(corr.detach().cpu().item())

    def _pism_diag_summarize_feature_matrix(self, features, prefix):
        """统计 PISM 输入矩阵分布。features shape: [valid_clients, input_dim]。"""
        summary = {
            f"{prefix}_names": self._pism_diag_input_names(),
        }
        if not torch.is_tensor(features) or features.dim() != 2:
            return summary

        features = features.detach().float()
        num_features = int(features.size(1))
        means = []
        stds = []
        mins = []
        maxs = []
        p10s = []
        p50s = []
        p90s = []
        zero_fracs = []
        nonfinite_fracs = []

        for feature_idx in range(num_features):
            value = features[:, feature_idx]
            finite_mask = torch.isfinite(value)
            nonfinite_frac = 1.0 - float(finite_mask.float().mean().detach().cpu().item())
            nonfinite_fracs.append(nonfinite_frac)

            if int(finite_mask.sum().item()) <= 0:
                means.append(None)
                stds.append(None)
                mins.append(None)
                maxs.append(None)
                p10s.append(None)
                p50s.append(None)
                p90s.append(None)
                zero_fracs.append(None)
                continue

            finite_value = value[finite_mask]
            means.append(float(finite_value.mean().detach().cpu().item()))
            stds.append(float(finite_value.std(unbiased=False).detach().cpu().item()))
            mins.append(float(finite_value.min().detach().cpu().item()))
            maxs.append(float(finite_value.max().detach().cpu().item()))
            p10s.append(float(torch.quantile(finite_value, 0.10).detach().cpu().item()))
            p50s.append(float(torch.quantile(finite_value, 0.50).detach().cpu().item()))
            p90s.append(float(torch.quantile(finite_value, 0.90).detach().cpu().item()))
            zero_fracs.append(float((finite_value.abs() <= 1e-12).float().mean().detach().cpu().item()))

        corr_matrix = []
        for row_idx in range(num_features):
            row = []
            for col_idx in range(num_features):
                row.append(
                    self._pism_diag_safe_pearson_corr(
                        features[:, row_idx],
                        features[:, col_idx],
                    )
                )
            corr_matrix.append(row)

        summary.update({
            f"{prefix}_mean": means,
            f"{prefix}_std": stds,
            f"{prefix}_min": mins,
            f"{prefix}_max": maxs,
            f"{prefix}_p10": p10s,
            f"{prefix}_p50": p50s,
            f"{prefix}_p90": p90s,
            f"{prefix}_zero_frac": zero_fracs,
            f"{prefix}_nonfinite_frac": nonfinite_fracs,
            f"{prefix}_corr_matrix": corr_matrix,
        })
        return summary

    def _pism_diag_summarize_feature_target_corr(self, features, target, prefix):
        """统计每个 PISM 输入和目标量之间的相关性。"""
        if not torch.is_tensor(features) or features.dim() != 2:
            return {f"{prefix}_corr": None}
        corr_values = []
        for feature_idx in range(features.size(1)):
            corr_values.append(
                self._pism_diag_safe_pearson_corr(features[:, feature_idx], target)
            )
        return {f"{prefix}_corr": corr_values}

    def _pism_diag_weight_kl(self, base_weights, alt_weights, eps=1e-12):
        """计算两个权重分布的 KL，用于输入消融敏感度诊断。"""
        if base_weights is None or alt_weights is None:
            return None
        base_weights = base_weights.detach().float().clamp_min(eps)
        alt_weights = alt_weights.detach().float().clamp_min(eps)
        value = (base_weights * (base_weights.log() - alt_weights.log())).sum()
        return self._pism_diag_safe_float(value)

    def _pism_diag_tensor_from_values(self, values, device=None):
        if values is None:
            return None
        try:
            if torch.is_tensor(values):
                tensor = values.detach()
                if device is not None:
                    tensor = tensor.to(device)
                return tensor.float().reshape(-1)
            return torch.tensor(values, device=device, dtype=torch.float32).reshape(-1)
        except (TypeError, ValueError, RuntimeError):
            return None

    def _pism_diag_normalize_weight_tensor(self, weights, eps=1e-12):
        weights = self._pism_diag_tensor_from_values(weights)
        if weights is None or weights.numel() == 0:
            return None
        if not torch.isfinite(weights).all():
            return None
        if (weights < 0).any():
            return None
        total = weights.sum()
        if (not torch.isfinite(total)) or total.item() <= eps:
            return None
        return weights / total

    def _pism_diag_eff_clients(self, weights, eps=1e-12):
        weights = self._pism_diag_normalize_weight_tensor(weights, eps=eps)
        if weights is None:
            return None
        denom = (weights * weights).sum()
        if (not torch.isfinite(denom)) or denom.item() <= eps:
            return None
        return self._pism_diag_safe_float(1.0 / denom)

    def _pism_diag_weight_gini(self, weights, eps=1e-12):
        weights = self._pism_diag_normalize_weight_tensor(weights, eps=eps)
        if weights is None:
            return None
        num_weights = int(weights.numel())
        if num_weights <= 1:
            return 0.0
        sorted_weights = torch.sort(weights).values
        index = torch.arange(
            1,
            num_weights + 1,
            device=sorted_weights.device,
            dtype=sorted_weights.dtype,
        )
        gini = (2.0 * (index * sorted_weights).sum() / num_weights) - (
            (num_weights + 1.0) / num_weights
        )
        return self._pism_diag_safe_float(gini.clamp(min=0.0, max=1.0))

    def _pism_diag_l1(self, weights_a, weights_b):
        weights_a = self._pism_diag_normalize_weight_tensor(weights_a)
        weights_b = self._pism_diag_normalize_weight_tensor(weights_b)
        if weights_a is None or weights_b is None or weights_a.numel() != weights_b.numel():
            return None
        return self._pism_diag_safe_float((weights_a - weights_b).abs().sum())

    def _pism_diag_kl(self, weights_a, weights_b, eps=1e-12):
        weights_a = self._pism_diag_normalize_weight_tensor(weights_a, eps=eps)
        weights_b = self._pism_diag_normalize_weight_tensor(weights_b, eps=eps)
        if weights_a is None or weights_b is None or weights_a.numel() != weights_b.numel():
            return None
        value = (weights_a * torch.log((weights_a + eps) / (weights_b + eps))).sum()
        return self._pism_diag_safe_float(value)

    def _pism_diag_rank_corr(self, weights_a, weights_b):
        weights_a = self._pism_diag_normalize_weight_tensor(weights_a)
        weights_b = self._pism_diag_normalize_weight_tensor(weights_b)
        if weights_a is None or weights_b is None or weights_a.numel() != weights_b.numel():
            return None
        if weights_a.numel() < 2:
            return None
        rank_dtype = torch.float32
        ranks = torch.arange(weights_a.numel(), device=weights_a.device, dtype=rank_dtype)
        rank_a = torch.empty(weights_a.numel(), device=weights_a.device, dtype=rank_dtype)
        rank_b = torch.empty(weights_b.numel(), device=weights_b.device, dtype=rank_dtype)
        rank_a[torch.argsort(weights_a, descending=True)] = ranks
        rank_b[torch.argsort(weights_b, descending=True)] = ranks.to(weights_b.device)
        return self._pism_diag_safe_pearson_corr(rank_a, rank_b)

    def _weighted_average_delta_states(self, delta_states_by_client, weights, device, eps=1e-12):
        if not isinstance(delta_states_by_client, dict) or not isinstance(weights, dict):
            return None
        selected = []
        for client_idx, weight in weights.items():
            weight = self._pism_diag_safe_float(weight)
            if weight is None or weight <= 0.0:
                continue
            delta_state = delta_states_by_client.get(client_idx)
            if not isinstance(delta_state, dict) or not delta_state:
                return None
            selected.append((client_idx, weight, delta_state))
        if not selected:
            return None

        total_weight = sum(weight for _, weight, _ in selected)
        if total_weight <= eps:
            return None
        keys = sorted(selected[0][2].keys())
        if not keys:
            return None
        expected_keys = set(keys)
        averaged_state = {}
        for _, _, delta_state in selected:
            if set(delta_state.keys()) != expected_keys:
                return None

        for key in keys:
            accumulator = None
            reference_shape = None
            for _, weight, delta_state in selected:
                tensor = delta_state.get(key)
                if tensor is None or not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
                    return None
                tensor = tensor.detach().to(device).float()
                if reference_shape is None:
                    reference_shape = tensor.shape
                    accumulator = torch.zeros_like(tensor)
                elif tensor.shape != reference_shape:
                    return None
                accumulator += (weight / total_weight) * tensor
            averaged_state[key] = accumulator
        return averaged_state

    def _pism_diag_delta_query_cos(self, delta_states_by_client, weights, query_grad_state, device):
        averaged_delta_state = self._weighted_average_delta_states(
            delta_states_by_client,
            weights,
            device,
        )
        if averaged_delta_state is None:
            return None
        return delta_to_negative_grad_score(
            averaged_delta_state,
            query_grad_state,
            metric="cosine",
            device=device,
        )

    def _add_pism_extra_diagnostics(
        self,
        metric,
        record,
        weights_tensor,
        logits_tensor,
        client_updates,
        global_state,
        device,
    ):
        valid_client_ids = list(record.get("valid_client_ids") or [])
        num_clients = len(valid_client_ids)
        if num_clients <= 0:
            return

        diag_device = weights_tensor.device if torch.is_tensor(weights_tensor) else device
        pism_weights = self._pism_diag_normalize_weight_tensor(weights_tensor)
        scores = self._pism_diag_tensor_from_values(record.get("scores"), device=diag_device)
        logits = self._pism_diag_tensor_from_values(logits_tensor, device=diag_device)
        sample_counts = self._pism_diag_tensor_from_values(
            record.get("client_sample_counts"),
            device=diag_device,
        )
        raw_losses = self._pism_diag_tensor_from_values(
            record.get("raw_client_losses"),
            device=diag_device,
        )
        raw_usages = self._pism_diag_tensor_from_values(
            record.get("raw_expert_usages"),
            device=diag_device,
        )
        raw_delta_norms = self._pism_diag_tensor_from_values(
            record.get("raw_delta_norms"),
            device=diag_device,
        )
        if pism_weights is None or scores is None or logits is None:
            return
        if (
            pism_weights.numel() != num_clients
            or scores.numel() != num_clients
            or logits.numel() != num_clients
        ):
            return

        def same_len(values):
            if values is None or values.numel() != num_clients:
                return None
            return values

        sample_counts = same_len(sample_counts)
        raw_losses = same_len(raw_losses)
        raw_usages = same_len(raw_usages)
        raw_delta_norms = same_len(raw_delta_norms)
        log_sample_counts = torch.log1p(sample_counts.clamp_min(0.0)) if sample_counts is not None else None
        log_usages = torch.log1p(raw_usages.clamp_min(0.0)) if raw_usages is not None else None
        log_delta_norms = torch.log1p(raw_delta_norms.clamp_min(0.0)) if raw_delta_norms is not None else None

        metric["pism_score_sample_corr"] = self._pism_diag_safe_pearson_corr(scores, log_sample_counts)
        metric["pism_weight_sample_corr"] = self._pism_diag_safe_pearson_corr(pism_weights, log_sample_counts)
        metric["pism_logit_sample_corr"] = self._pism_diag_safe_pearson_corr(logits, log_sample_counts)
        metric["pism_delta_norm_sample_corr"] = self._pism_diag_safe_pearson_corr(log_delta_norms, log_sample_counts)
        metric["pism_usage_sample_corr"] = self._pism_diag_safe_pearson_corr(log_usages, log_sample_counts)

        metric["pism_weight_usage_corr"] = self._pism_diag_safe_pearson_corr(pism_weights, log_usages)
        metric["pism_weight_delta_norm_corr"] = self._pism_diag_safe_pearson_corr(pism_weights, log_delta_norms)
        metric["pism_weight_loss_corr"] = self._pism_diag_safe_pearson_corr(pism_weights, raw_losses)
        metric["pism_logit_usage_corr"] = self._pism_diag_safe_pearson_corr(logits, log_usages)
        metric["pism_logit_delta_norm_corr"] = self._pism_diag_safe_pearson_corr(logits, log_delta_norms)
        metric["pism_logit_loss_corr"] = self._pism_diag_safe_pearson_corr(logits, raw_losses)

        metric["pism_weight_top1"] = self._pism_diag_safe_float(pism_weights.max())
        top2_values = torch.topk(pism_weights, k=min(2, int(pism_weights.numel()))).values
        metric["pism_weight_top2_sum"] = self._pism_diag_safe_float(top2_values.sum())
        metric["pism_weight_eff_clients"] = self._pism_diag_eff_clients(pism_weights)
        metric["pism_weight_gini"] = self._pism_diag_weight_gini(pism_weights)

        client_score_dict = {
            client_idx: float(score.detach().cpu().item())
            for client_idx, score in zip(valid_client_ids, scores)
        }
        foga_weight_dict, _ = positive_score_to_weights(
            client_score_dict,
            mode=getattr(self.args, "uoc_foga_score_mode", "relu"),
        )
        foga_weights = None
        if foga_weight_dict:
            foga_weights = self._pism_diag_tensor_from_values(
                [foga_weight_dict.get(client_idx, 0.0) for client_idx in valid_client_ids],
                device=diag_device,
            )
            foga_weights = self._pism_diag_normalize_weight_tensor(foga_weights)

        if torch.isfinite(scores).all() and pism_weights.numel() > 0:
            foga_top_pos = int(torch.argmax(scores).detach().cpu().item())
            pism_top_pos = int(torch.argmax(pism_weights).detach().cpu().item())
            metric["pism_top_minus_foga_top_score"] = self._pism_diag_safe_float(
                scores[pism_top_pos] - scores[foga_top_pos]
            )

            def set_top_gap(key, values):
                if values is None:
                    return
                metric[key] = self._pism_diag_safe_float(
                    values[pism_top_pos] - values[foga_top_pos]
                )

            set_top_gap("pism_top_minus_foga_top_usage", log_usages)
            set_top_gap("pism_top_minus_foga_top_delta_norm", log_delta_norms)
            set_top_gap("pism_top_minus_foga_top_sample_size", log_sample_counts)

        if foga_weights is not None and foga_weights.numel() == num_clients:
            metric["pism_foga_weight_l1"] = self._pism_diag_l1(pism_weights, foga_weights)
            metric["pism_foga_weight_kl"] = self._pism_diag_kl(pism_weights, foga_weights)
            metric["pism_foga_top_match"] = float(
                int(
                    torch.argmax(pism_weights).detach().cpu().item()
                    == torch.argmax(foga_weights).detach().cpu().item()
                )
            )
            metric["pism_foga_rank_corr"] = self._pism_diag_rank_corr(pism_weights, foga_weights)

        delta_states_by_client = {}
        for client_idx in valid_client_ids:
            if client_idx < 0 or client_idx >= len(client_updates):
                continue
            delta_state = extract_expert_delta_state(
                client_updates[client_idx],
                global_state,
                record["expert_keys"],
                device=device,
            )
            if delta_state:
                delta_states_by_client[client_idx] = delta_state

        pism_weight_dict = {
            client_idx: float(weight.detach().cpu().item())
            for client_idx, weight in zip(valid_client_ids, pism_weights)
        }
        uniform_weight_dict = {
            client_idx: 1.0 / num_clients
            for client_idx in valid_client_ids
        }
        metric["pism_agg_delta_query_cos"] = self._pism_diag_delta_query_cos(
            delta_states_by_client,
            pism_weight_dict,
            record.get("query_grad_state"),
            device,
        )
        if foga_weights is not None and foga_weights.numel() == num_clients:
            foga_weight_dict_for_diag = {
                client_idx: float(weight.detach().cpu().item())
                for client_idx, weight in zip(valid_client_ids, foga_weights)
            }
            metric["foga_agg_delta_query_cos"] = self._pism_diag_delta_query_cos(
                delta_states_by_client,
                foga_weight_dict_for_diag,
                record.get("query_grad_state"),
                device,
            )
        metric["uniform_agg_delta_query_cos"] = self._pism_diag_delta_query_cos(
            delta_states_by_client,
            uniform_weight_dict,
            record.get("query_grad_state"),
            device,
        )
        pism_cos = metric.get("pism_agg_delta_query_cos")
        foga_cos = metric.get("foga_agg_delta_query_cos")
        uniform_cos = metric.get("uniform_agg_delta_query_cos")
        if pism_cos is not None and uniform_cos is not None:
            metric["pism_vs_uniform_agg_delta_query_cos_gap"] = float(pism_cos - uniform_cos)
        if pism_cos is not None and foga_cos is not None:
            metric["pism_vs_foga_agg_delta_query_cos_gap"] = float(pism_cos - foga_cos)

    def _pism_diag_score_alignment_loss(self, features, scores, tau=1.0, eps=1e-12):
        """只用于诊断的 score 对齐损失，不参与 optimizer.step，不改变训练随机性。"""
        if features is None or scores is None:
            return None
        if not torch.is_tensor(features) or not torch.is_tensor(scores):
            return None
        if features.numel() == 0 or scores.numel() == 0:
            return None

        was_training = self.meta_net.training
        try:
            # 诊断 forward 固定用 eval，避免 dropout 消耗随机数状态，保证不影响真正训练逻辑。
            self.meta_net.eval()
            with torch.no_grad():
                weights = self.meta_net(features, tau=tau)
                target = torch.softmax(
                    scores.detach().float() / max(float(tau), eps),
                    dim=0,
                )
                loss = -(target * torch.log(weights.clamp_min(eps))).sum()
            return self._pism_diag_safe_float(loss)
        finally:
            self.meta_net.train(was_training)

    def _pism_diag_feature_grad_abs_mean(self, features, scores, tau=1.0, eps=1e-12):
        """计算诊断损失对每个输入维度的梯度均值。"""
        if features is None or scores is None:
            return None
        if not torch.is_tensor(features) or not torch.is_tensor(scores):
            return None
        if features.numel() == 0 or scores.numel() == 0:
            return None

        was_training = self.meta_net.training
        try:
            self.meta_net.eval()
            diag_features = features.detach().clone().requires_grad_(True)
            weights = self.meta_net(diag_features, tau=tau)
            target = torch.softmax(scores.detach().float() / max(float(tau), eps), dim=0)
            loss = -(target * torch.log(weights.clamp_min(eps))).sum()
            grad = torch.autograd.grad(
                loss,
                diag_features,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                return None
            return [
                float(value)
                for value in grad.detach().abs().mean(dim=0).cpu().tolist()
            ]
        finally:
            self.meta_net.train(was_training)

    def _pism_diag_first_layer_weight_norm_by_input(self):
        """统计 PISM encoder 第一层对每个输入维度的参数范数。"""
        first_linear = None
        encoder = getattr(self.meta_net, "encoder", None)
        if encoder is not None:
            for module in encoder.modules():
                if isinstance(module, torch.nn.Linear):
                    first_linear = module
                    break
        if first_linear is None:
            return None
        weight = first_linear.weight.detach().float()
        return [
            float(value)
            for value in weight.norm(dim=0).cpu().tolist()
        ]

    def _pism_diag_first_layer_grad_norm_by_input(self):
        """统计 PISM encoder 第一层当前梯度在每个输入维度上的范数。"""
        first_linear = None
        encoder = getattr(self.meta_net, "encoder", None)
        if encoder is not None:
            for module in encoder.modules():
                if isinstance(module, torch.nn.Linear):
                    first_linear = module
                    break
        if first_linear is None or first_linear.weight.grad is None:
            return None
        grad = first_linear.weight.grad.detach().float()
        return [
            float(value)
            for value in grad.norm(dim=0).cpu().tolist()
        ]

    def _pism_diag_mean_scalar_metric(self, expert_metrics, key):
        """对 per-expert 标量诊断取均值。"""
        values = []
        for metric in expert_metrics:
            value = self._pism_diag_safe_float(metric.get(key))
            if value is not None:
                values.append(value)
        if not values:
            return None
        return sum(values) / len(values)

    def _pism_diag_mean_vector_metric(self, expert_metrics, key):
        """对 per-expert 向量诊断逐维取均值。"""
        sums = None
        counts = None
        for metric in expert_metrics:
            value = metric.get(key)
            if not isinstance(value, (list, tuple)):
                continue
            if sums is None:
                sums = [0.0 for _ in range(len(value))]
                counts = [0 for _ in range(len(value))]
            for idx, item in enumerate(value):
                if idx >= len(sums):
                    continue
                item = self._pism_diag_safe_float(item)
                if item is None:
                    continue
                sums[idx] += item
                counts[idx] += 1
        if sums is None:
            return None
        return [
            sums[idx] / counts[idx] if counts[idx] > 0 else None
            for idx in range(len(sums))
        ]

    def _build_pism_record_for_expert(
        self,
        aggregated_state,
        global_state,
        client_updates,
        client_stats,
        client_sample_counts,
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
        delta_norms = []
        client_sample_counts_for_valid = []

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
            client_losses.append(self._get_client_loss_from_stats(client_stat))
            expert_usages.append(
                self._get_expert_usage_from_stats(client_stat, layer_id, expert_id)
            )
            delta_norms.append(l2_norm_state(delta_state, device=device))
            sample_count = self._get_indexed_value(client_sample_counts, client_idx)
            client_sample_counts_for_valid.append(
                float(sample_count) if sample_count is not None else float("nan")
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

        features = build_pism_feature_tensor(
            client_loss=client_losses,
            expert_usage=expert_usages,
            delta_norm=delta_norms,
            device=device,
        )
        if score_metric in {"cosine", "delta_consensus"} and self.pism_input_dim == 3:
            expert_usage = torch.as_tensor(
                expert_usages,
                device=device,
                dtype=torch.float32,
            ).reshape(-1)
            sample_count = torch.as_tensor(
                client_sample_counts_for_valid,
                device=device,
                dtype=torch.float32,
            ).reshape(-1)
            if expert_usage.numel() != sample_count.numel() or features.size(0) != sample_count.numel():
                raise ValueError("PISM usage ratio inputs must have the same length")
            expert_usage_ratio = expert_usage / sample_count.clamp_min(1.0)
            features = torch.stack(
                [
                    features[..., 0],
                    features[..., 1],
                    expert_usage_ratio,
                    features[..., 2],
                ],
                dim=-1,
            )
            features = self._select_pism_features_for_config(features)
        scores_tensor = torch.tensor(scores, device=device, dtype=torch.float32).detach()

        # 归一化前的输入诊断：只读 features，不影响算法。
        metric["pism_input_names"] = self._pism_diag_input_names()
        metric.update(
            self._pism_diag_summarize_feature_matrix(
                features,
                prefix="pism_raw_input",
            )
        )
        metric.update(
            self._pism_diag_summarize_feature_target_corr(
                features,
                scores_tensor,
                prefix="pism_raw_input_score",
            )
        )

        if self.pism_renorm_inputs:
            features = normalize_pism_inputs(features)

        # 归一化后的输入诊断：这里的 features 才是真正喂给 PISM 的输入。
        metric.update(
            self._pism_diag_summarize_feature_matrix(
                features,
                prefix="pism_norm_input",
            )
        )
        metric.update(
            self._pism_diag_summarize_feature_target_corr(
                features,
                scores_tensor,
                prefix="pism_norm_input_score",
            )
        )
        if not torch.isfinite(features).all():
            return metric, self._fallback_expert(
                aggregated_state,
                global_state,
                client_updates,
                expert_keys,
                metric,
                "pism_features_nan",
            )
        feature_std_for_diag = features.detach().float().std(dim=0, unbiased=False)
        metric["pism_feature_collapse_frac"] = float(
            (feature_std_for_diag <= 1e-8).float().mean().detach().cpu().item()
        )
        tau_for_diag = max(float(self.pism_tau), 1e-12)
        metric["pism_diag_alignment_loss_before_step"] = self._pism_diag_score_alignment_loss(
            features,
            scores_tensor,
            tau=tau_for_diag,
        )
        metric["pism_feature_grad_abs_mean"] = self._pism_diag_feature_grad_abs_mean(
            features,
            scores_tensor,
            tau=tau_for_diag,
        )
        metric["pism_first_layer_weight_norm_by_input"] = (
            self._pism_diag_first_layer_weight_norm_by_input()
        )



        # 兼容旧日志字段。
        metric["pism_input_mean"] = [
            float(value) for value in features.detach().mean(dim=0).cpu().tolist()
        ]
        metric["pism_input_std"] = [
            float(value)
            for value in features.detach().std(dim=0, unbiased=False).cpu().tolist()
        ]

        record = {
            "layer_id": str(layer_id),
            "expert_id": str(expert_id),
            "expert_keys": expert_keys,
            "valid_client_ids": valid_client_ids,
            "features": features,
            "scores": scores_tensor,
            "raw_client_losses": client_losses,
            "raw_expert_usages": expert_usages,
            "raw_delta_norms": delta_norms,
            "client_sample_counts": client_sample_counts_for_valid,
            "query_grad_state": grad_state,
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
        tau = self.pism_tau

        with torch.no_grad():
            pism_output = self.meta_net(
                record["features"],
                tau=tau,
                return_logits=True,
            )
            weights_tensor = pism_output["weights"]
            logits_tensor = pism_output["logits"]

            metric["pism_logits_mean"] = float(logits_tensor.mean().detach().cpu().item())
            metric["pism_logits_std"] = float(
                logits_tensor.std(unbiased=False).detach().cpu().item()
            )
            metric["pism_logits_min"] = float(logits_tensor.min().detach().cpu().item())
            metric["pism_logits_max"] = float(logits_tensor.max().detach().cpu().item())

            metric["pism_weight_score_corr"] = self._pism_diag_safe_pearson_corr(
                weights_tensor,
                record["scores"],
            )
            metric.update(
                self._pism_diag_summarize_feature_target_corr(
                    record["features"],
                    weights_tensor,
                    prefix="pism_weight_input",
                )
            )
            metric.update(
                self._pism_diag_summarize_feature_target_corr(
                    record["features"],
                    logits_tensor,
                    prefix="pism_logit_input",
                )
            )

            metric["pism_diag_alignment_loss_after_step"] = self._pism_diag_score_alignment_loss(
                record["features"],
                record["scores"],
                tau=tau,
            )
            before_loss = metric.get("pism_diag_alignment_loss_before_step")
            after_loss = metric.get("pism_diag_alignment_loss_after_step")
            if before_loss is None or after_loss is None:
                metric["pism_diag_alignment_loss_delta"] = None
            else:
                metric["pism_diag_alignment_loss_delta"] = float(before_loss - after_loss)

            score_top_pos = int(torch.argmax(record["scores"]).detach().cpu().item())
            weight_top_pos = int(torch.argmax(weights_tensor).detach().cpu().item())
            metric["pism_top_score_client_id"] = int(record["valid_client_ids"][score_top_pos])
            metric["pism_top_weight_client_id"] = int(record["valid_client_ids"][weight_top_pos])
            metric["pism_top_weight_matches_top_score"] = int(score_top_pos == weight_top_pos)

            feature_sensitivity_l1 = []
            feature_sensitivity_kl = []
            feature_sensitivity_top_change = []
            for feature_idx in range(record["features"].size(-1)):
                ablated_features = record["features"].detach().clone()
                # renorm 后 0 代表该特征处在本 expert 的均值位置。
                ablated_features[:, feature_idx] = 0.0
                ablated_output = self.meta_net(
                    ablated_features,
                    tau=tau,
                    return_logits=True,
                )
                ablated_weights = ablated_output["weights"]
                sensitivity_l1 = (ablated_weights - weights_tensor).abs().mean()
                feature_sensitivity_l1.append(float(sensitivity_l1.detach().cpu().item()))
                feature_sensitivity_kl.append(
                    self._pism_diag_weight_kl(weights_tensor, ablated_weights)
                )
                ablated_top_pos = int(torch.argmax(ablated_weights).detach().cpu().item())
                feature_sensitivity_top_change.append(int(ablated_top_pos != weight_top_pos))

            metric["pism_feature_sensitivity_l1"] = feature_sensitivity_l1
            metric["pism_feature_sensitivity_kl"] = feature_sensitivity_kl
            metric["pism_feature_sensitivity_top_change"] = feature_sensitivity_top_change

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
        self._add_pism_extra_diagnostics(
            metric,
            record,
            weights_tensor,
            logits_tensor,
            client_updates,
            global_state,
            device,
        )
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
                client_sample_counts=client_weights,
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
        if per_expert_records:
            tau = self.pism_tau
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

                # 只读当前 backward 后的第一层梯度范数，不 step、不改训练逻辑。
                first_layer_grad_norm = self._pism_diag_first_layer_grad_norm_by_input()
                for record in per_expert_records:
                    record["metric"]["pism_first_layer_grad_norm_by_input"] = first_layer_grad_norm

                self.meta_optimizer.step()
                self.pism_update_steps += 1
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
        pism_success_metrics = [
            expert_metric
            for expert_metric in expert_metrics
            if bool(expert_metric.get("pism_used", False))
            and expert_metric.get("fallback_reason") is None
            and expert_metric.get("pism_fallback_reason") is None
        ]
        pism_summary = {
            "uoc_foga_pism_meta_loss_mean": (
                sum(pism_meta_losses) / len(pism_meta_losses)
                if pism_meta_losses else None
            ),
            "uoc_foga_pism_updated_experts": updated_experts,
            "uoc_foga_pism_fallback_experts": fallback_experts,
            "uoc_foga_pism_fallback_reason_counts": dict(pism_fallback_reason_counts),
            "uoc_foga_pism_weight_entropy_mean": (
                sum(pism_weight_entropies) / len(pism_weight_entropies)
                if pism_weight_entropies else None
            ),
            "uoc_foga_pism_weight_max_mean": (
                sum(pism_weight_max_values) / len(pism_weight_max_values)
                if pism_weight_max_values else None
            ),
            "uoc_foga_pism_used_frac": (
                updated_experts / total_experts if total_experts > 0 else 0.0
            ),
            "uoc_foga_pism_update_steps": int(self.pism_update_steps),
            # PISM 三输入诊断 summary。
            "uoc_foga_pism_input_names": self._pism_diag_input_names(),
            "uoc_foga_pism_raw_input_std_mean": self._pism_diag_mean_vector_metric(
                expert_metrics,
                "pism_raw_input_std",
            ),
            "uoc_foga_pism_norm_input_std_mean": self._pism_diag_mean_vector_metric(
                expert_metrics,
                "pism_norm_input_std",
            ),
            "uoc_foga_pism_raw_input_score_corr_mean": self._pism_diag_mean_vector_metric(
                expert_metrics,
                "pism_raw_input_score_corr",
            ),
            "uoc_foga_pism_norm_input_score_corr_mean": self._pism_diag_mean_vector_metric(
                expert_metrics,
                "pism_norm_input_score_corr",
            ),
            "uoc_foga_pism_weight_input_corr_mean": self._pism_diag_mean_vector_metric(
                expert_metrics,
                "pism_weight_input_corr",
            ),
            "uoc_foga_pism_logit_input_corr_mean": self._pism_diag_mean_vector_metric(
                expert_metrics,
                "pism_logit_input_corr",
            ),
            "uoc_foga_pism_feature_sensitivity_l1_mean": self._pism_diag_mean_vector_metric(
                expert_metrics,
                "pism_feature_sensitivity_l1",
            ),
            "uoc_foga_pism_feature_sensitivity_kl_mean": self._pism_diag_mean_vector_metric(
                expert_metrics,
                "pism_feature_sensitivity_kl",
            ),
            "uoc_foga_pism_feature_sensitivity_top_change_frac": self._pism_diag_mean_vector_metric(
                expert_metrics,
                "pism_feature_sensitivity_top_change",
            ),
            "uoc_foga_pism_feature_grad_abs_mean": self._pism_diag_mean_vector_metric(
                expert_metrics,
                "pism_feature_grad_abs_mean",
            ),
            "uoc_foga_pism_first_layer_weight_norm_by_input": self._pism_diag_mean_vector_metric(
                expert_metrics,
                "pism_first_layer_weight_norm_by_input",
            ),
            "uoc_foga_pism_first_layer_grad_norm_by_input": self._pism_diag_mean_vector_metric(
                expert_metrics,
                "pism_first_layer_grad_norm_by_input",
            ),
            "uoc_foga_pism_weight_score_corr_mean": self._pism_diag_mean_scalar_metric(
                expert_metrics,
                "pism_weight_score_corr",
            ),
            "uoc_foga_pism_top_weight_match_score_frac": self._pism_diag_mean_scalar_metric(
                expert_metrics,
                "pism_top_weight_matches_top_score",
            ),
            "uoc_foga_pism_feature_collapse_frac_mean": self._pism_diag_mean_scalar_metric(
                expert_metrics,
                "pism_feature_collapse_frac",
            ),
            "uoc_foga_pism_logits_std_mean": self._pism_diag_mean_scalar_metric(
                expert_metrics,
                "pism_logits_std",
            ),
            "uoc_foga_pism_diag_alignment_loss_before_step_mean": self._pism_diag_mean_scalar_metric(
                expert_metrics,
                "pism_diag_alignment_loss_before_step",
            ),
            "uoc_foga_pism_diag_alignment_loss_after_step_mean": self._pism_diag_mean_scalar_metric(
                expert_metrics,
                "pism_diag_alignment_loss_after_step",
            ),
            "uoc_foga_pism_diag_alignment_loss_delta_mean": self._pism_diag_mean_scalar_metric(
                expert_metrics,
                "pism_diag_alignment_loss_delta",
            ),
            "uoc_foga_pism_score_sample_corr_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_score_sample_corr",
            ),
            "uoc_foga_pism_weight_sample_corr_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_weight_sample_corr",
            ),
            "uoc_foga_pism_logit_sample_corr_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_logit_sample_corr",
            ),
            "uoc_foga_pism_delta_norm_sample_corr_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_delta_norm_sample_corr",
            ),
            "uoc_foga_pism_usage_sample_corr_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_usage_sample_corr",
            ),
            "uoc_foga_pism_weight_usage_corr_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_weight_usage_corr",
            ),
            "uoc_foga_pism_weight_delta_norm_corr_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_weight_delta_norm_corr",
            ),
            "uoc_foga_pism_weight_loss_corr_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_weight_loss_corr",
            ),
            "uoc_foga_pism_logit_usage_corr_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_logit_usage_corr",
            ),
            "uoc_foga_pism_logit_delta_norm_corr_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_logit_delta_norm_corr",
            ),
            "uoc_foga_pism_logit_loss_corr_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_logit_loss_corr",
            ),
            "uoc_foga_pism_agg_delta_query_cos_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_agg_delta_query_cos",
            ),
            "uoc_foga_foga_agg_delta_query_cos_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "foga_agg_delta_query_cos",
            ),
            "uoc_foga_uniform_agg_delta_query_cos_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "uniform_agg_delta_query_cos",
            ),
            "uoc_foga_pism_vs_uniform_agg_delta_query_cos_gap_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_vs_uniform_agg_delta_query_cos_gap",
            ),
            "uoc_foga_pism_vs_foga_agg_delta_query_cos_gap_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_vs_foga_agg_delta_query_cos_gap",
            ),
            "uoc_foga_pism_weight_top1_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_weight_top1",
            ),
            "uoc_foga_pism_weight_top2_sum_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_weight_top2_sum",
            ),
            "uoc_foga_pism_weight_eff_clients_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_weight_eff_clients",
            ),
            "uoc_foga_pism_weight_gini_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_weight_gini",
            ),
            "uoc_foga_pism_top_minus_foga_top_score_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_top_minus_foga_top_score",
            ),
            "uoc_foga_pism_top_minus_foga_top_usage_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_top_minus_foga_top_usage",
            ),
            "uoc_foga_pism_top_minus_foga_top_delta_norm_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_top_minus_foga_top_delta_norm",
            ),
            "uoc_foga_pism_top_minus_foga_top_sample_size_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_top_minus_foga_top_sample_size",
            ),
            "uoc_foga_pism_foga_weight_l1_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_foga_weight_l1",
            ),
            "uoc_foga_pism_foga_weight_kl_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_foga_weight_kl",
            ),
            "uoc_foga_pism_foga_top_match_frac": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_foga_top_match",
            ),
            "uoc_foga_pism_foga_rank_corr_mean": self._pism_diag_mean_scalar_metric(
                pism_success_metrics,
                "pism_foga_rank_corr",
            ),
        }
        self.last_aggregation_metrics = {
            "uoc_foga_stats": uoc_foga_stats,
            "uoc_foga_pism_summary": pism_summary,
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
