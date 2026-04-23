import collections
import copy
import math
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

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
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

        updated_bayes_state["round"] = int(updated_bayes_state.get("round", 0)) + 1

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
        )
        return aggregated_params, len(client_payloads), local_posterior_count, expert_metric

    def _optimize_expert_prior(
        self,
        expert_keys,
        global_state,
        prior_state,
        prior_n0,
        client_payloads,
    ):
        prior_mean_params = collections.OrderedDict()
        log_precision_params = collections.OrderedDict()
        optim_params = []

        for key in expert_keys:
            mean_param = torch.nn.Parameter(
                global_state[key].detach().cpu().clone().to(dtype=torch.float32)
            )
            prior_mean_params[key] = mean_param
            optim_params.append(mean_param)

            reference_tensor = global_state[key].detach().cpu().clone().to(dtype=torch.float32)
            log_precision = prior_state.get("log_precision_state", {}).get(key)
            if log_precision is None or not torch.is_floating_point(log_precision):
                log_precision = torch.full_like(reference_tensor, fill_value=math.log(self.gamma0_init))
            else:
                log_precision = log_precision.detach().cpu().clone().to(dtype=torch.float32)

            log_precision_param = torch.nn.Parameter(
                log_precision,
                requires_grad=self.update_precision,
            )
            log_precision_params[key] = log_precision_param
            if self.update_precision:
                optim_params.append(log_precision_param)

        log_n0_value = prior_state.get("log_n0")
        if log_n0_value is None:
            log_n0_value = torch.tensor(math.log(max(float(prior_n0.item()), self.min_precision)), dtype=torch.float32)
        else:
            log_n0_value = log_n0_value.detach().cpu().clone().to(dtype=torch.float32)
        log_n0_param = torch.nn.Parameter(
            log_n0_value,
            requires_grad=self.update_strength,
        )
        if self.update_strength:
            optim_params.append(log_n0_param)

        optimizer = torch.optim.Adam(optim_params, lr=self.meta_lr)
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
            meta_loss.backward()
            optimizer.step()
            self._project_meta_params(log_precision_params, log_n0_param)

        with torch.no_grad():
            final_meta_loss, final_local_posterior_count = self._compute_expert_meta_loss(
                expert_keys=expert_keys,
                prior_mean_params=prior_mean_params,
                log_precision_params=log_precision_params,
                log_n0_param=log_n0_param,
                client_payloads=client_payloads,
            )

        return (
            prior_mean_params,
            log_precision_params,
            log_n0_param,
            final_local_posterior_count,
            float(final_meta_loss.detach().cpu().item()),
        )

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

            # 按算法文档 Step 5：
            #   |S_k|^{-1} * sum_i [ f_{i,k}(L_{0,k}) + 1/2 g_{i,k}(L_{0,k}) ]
            # 这里应当直接对“每个客户端的总项”做平均，
            # 不再额外按参数维度做归一化。
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
            avg_gamma0 = float(torch.cat(gamma_values).mean().item())
        else:
            avg_gamma0 = float(self.gamma0_init)

        if optimized_log_n0 is None:
            log_n0 = prior_state.get("log_n0")
            if log_n0 is None:
                n0 = float(self.n0_init)
            else:
                n0 = float(torch.exp(log_n0.detach().cpu().float()).item())
        else:
            n0 = float(torch.exp(optimized_log_n0.detach().cpu().float()).item())

        return {
            "status": status,
            "layer_id": str(layer_id),
            "expert_id": str(expert_id),
            "clients": int(contributing_clients),
            "local_posteriors": int(local_posterior_count),
            "meta_loss": None if meta_loss is None else round(float(meta_loss), 6),
            "n0": round(n0, 6),
            "avg_gamma0": round(avg_gamma0, 6),
        }

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
                    "mean_state": evidence.get("mean_state", {}),
                    "precision_state": evidence.get("precision_state", {}),
                }
            )
        return payloads

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
