import math
import torch
import torch.optim as optim
import time
from types import SimpleNamespace
from torch import nn

from data.loader import build_client_train_loader
from fl.bayes_utils import run_expert_sgld_fit
from model import build_model_from_args
from utils.utils import record_result

class Client:
    # Client 表示联邦学习里的一个客户端。
    # 每个客户端有自己的数据和模型，服务端每一轮会让多个客户端分别训练。
    def __init__(
        self,
        args: SimpleNamespace,
        client_id: int,
        logger,
        c_T: int,
        partition_meta=None,
        server_state_dict=None,
    ):
        self.args = args
        self.client_id = client_id
        self.server_state_dict = server_state_dict
        self.save_client_models = bool(getattr(self.args, "save_client_models", False))
        # 从 save/model/{client_id}.pth 加载这个客户端自己的模型。
        self.model = self.load_client_model()
        self.device = self.args.device
        self.model.to(self.device)
        # c_T 表示当前是第几轮服务端通信轮次，主要用于记录日志。
        self.c_T =  c_T
        self.client_epochs = self.args.client_epochs
        # 分类任务常用交叉熵损失。
        self.criterion = nn.CrossEntropyLoss()
        self.current_learning_rate = self.get_current_learning_rate()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.current_learning_rate)

        self.batch_size = self.args.batch_size
        self.partition_meta = partition_meta
        self.train_loader = None
        # 加载当前客户端的训练索引，并动态封装成 DataLoader。
        self.get_dataloader()

        self.logger = logger
        self.router_aux_loss_coef = self.args.router_aux_loss_coef
        self.router_z_loss_coef = self.args.router_z_loss_coef
        self.bayes_min_expert_tokens = getattr(self.args, "bayes_min_expert_tokens", 1)
        self.bayes_sgld_steps = max(int(getattr(self.args, "bayes_sgld_steps", 20)), 1)
        self.bayes_sgld_burnin = min(
            max(int(getattr(self.args, "bayes_sgld_burnin", 10)), 0),
            self.bayes_sgld_steps - 1,
        )
        self.bayes_sgld_lr = float(getattr(self.args, "bayes_sgld_lr", 0.00005))
        self.bayes_ai_max = float(getattr(self.args, "bayes_ai_max", 1e3))
        self.bayes_sgld_var_floor = max(float(getattr(self.args, "bayes_sgld_var_floor", 0.0)), 0.0)
        self.bayes_precision_mode = getattr(self.args, "bayes_precision_mode", "floor_inverse")
        self.bayes_precision_temperature = float(getattr(self.args, "bayes_precision_temperature", 0.25))
        self.bayes_precision_target = float(getattr(self.args, "bayes_precision_target", 100.0))
        self.bayes_precision_min = float(getattr(self.args, "bayes_precision_min", 20.0))
        self.bayes_precision_max = float(getattr(self.args, "bayes_precision_max", 300.0))
        self.bayes_precision_eps = float(getattr(self.args, "bayes_precision_eps", 1.0e-12))
        self.bayes_evidence_batches = max(int(getattr(self.args, "bayes_evidence_batches", 8)), 1)
        self.bayes_evidence_log_detail = bool(getattr(self.args, "bayes_evidence_log_detail", False))
        self.bayes_sgld_concat_cache = bool(getattr(self.args, "bayes_sgld_concat_cache", False))
        self.bayes_empty_cache_after_client_evidence = bool(
            getattr(self.args, "bayes_empty_cache_after_client_evidence", False)
        )
        self.bayes_cache_device = str(getattr(self.args, "bayes_cache_device", "cpu")).lower()
        if self.bayes_cache_device not in {"cpu", "cuda", "auto"}:
            raise ValueError("bayes_cache_device must be one of: cpu, cuda, auto")

    def get_current_learning_rate(self):
        base_lr = float(self.args.learning_rate)
        scheduler = str(getattr(self.args, "lr_scheduler", "none")).lower()
        total_rounds = max(int(getattr(self.args, "server_epochs", 1)), 1)
        round_idx = min(max(int(self.c_T), 0), total_rounds - 1)

        lr_min_value = getattr(self.args, "lr_min", None)
        lr_min = base_lr * 0.1 if lr_min_value is None else float(lr_min_value)
        warmup_rounds = max(int(getattr(self.args, "lr_warmup_rounds", 0)), 0)
        warmup_start_value = getattr(self.args, "lr_warmup_start_lr", None)
        warmup_start_lr = lr_min if warmup_start_value is None else float(warmup_start_value)

        if scheduler in {"none", "constant", "off"}:
            return base_lr

        if scheduler == "cosine":
            if total_rounds <= 1:
                return base_lr
            progress = round_idx / (total_rounds - 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min + (base_lr - lr_min) * cosine

        if scheduler == "cosine_warmup":
            warmup_rounds = min(warmup_rounds, total_rounds)
            if warmup_rounds > 0 and round_idx < warmup_rounds:
                if warmup_rounds == 1:
                    return base_lr
                warmup_progress = round_idx / (warmup_rounds - 1)
                return warmup_start_lr + (base_lr - warmup_start_lr) * warmup_progress

            remaining_rounds = total_rounds - warmup_rounds
            if remaining_rounds <= 1:
                return base_lr if remaining_rounds <= 0 else lr_min

            cosine_idx = round_idx - warmup_rounds
            progress = cosine_idx / (remaining_rounds - 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min + (base_lr - lr_min) * cosine

        raise ValueError(
            f"Unsupported lr_scheduler: {scheduler!r}. "
            "Expected one of: none, constant, off, cosine, cosine_warmup."
        )

    def load_client_model(self):
        # 客户端模型路径，例如 ./save/model/1.pth。
        self.model_path = self.args.model_save_path + f"/{self.client_id}.pth"
        model = build_model_from_args(self.args)
        if self.save_client_models:
            state_dict = torch.load(self.model_path, map_location="cpu")
            model.load_state_dict(state_dict)
        return model

    def save_client_model(self, state_dict=None):
        # 本地训练结束后，把客户端模型保存回原来的路径。
        if state_dict is None:
            state_dict = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }
        torch.save(state_dict, self.model_path)

    def get_dataloader(self):
        # 客户端只拥有自己的训练数据；验证和测试都由服务端统一执行。
        self.train_loader = build_client_train_loader(
            args=self.args,
            client_id=self.client_id,
            meta=self.partition_meta,
        )


    def renew_model(self):
        # 每一轮本地训练前，客户端同步服务端 state_dict。
        if self.server_state_dict is not None:
            server_state_dict = self.server_state_dict
        else:
            server_state_dict = torch.load(
                self.args.model_save_path + f"/server.pth",
                map_location="cpu",
            )
        self.model.load_state_dict(server_state_dict)

    def get_auxiliary_losses(self, result):
        zero = torch.tensor(0.0, device=self.device)
        router_aux_loss = result.get("router_aux_loss", result.get("aux_loss", zero))
        router_z_loss = result.get("router_z_loss", zero)
        extra_loss = (
            self.router_aux_loss_coef * router_aux_loss
            + self.router_z_loss_coef * router_z_loss
        )
        return extra_loss, router_aux_loss, router_z_loss

    def get_expert_activations(self, result):
        usage = result.get("expert_activations")
        if usage is None:
            usage = torch.zeros(self.args.num_experts, device=self.device)
        return usage.to(self.device)

    def get_avg_router_probs(self, result):
        probs = result.get("avg_router_probs")
        if probs is None:
            probs = torch.zeros(self.args.num_experts, device=self.device)
        return probs.to(self.device)

    def get_layer_expert_stats(self, result):
        layer_stats = result.get("expert_stats_by_layer")
        if layer_stats is not None:
            return layer_stats

        return {
            layer_id: {"expert_activations": usage}
            for layer_id, usage in result.get("expert_activations_by_layer", {}).items()
        }

    def add_layer_stats(self, total_stats, batch_stats):
        for layer_id, stats in batch_stats.items():
            layer_key = str(layer_id)
            if layer_key not in total_stats:
                total_stats[layer_key] = {
                    "expert_activations": torch.zeros(self.args.num_experts, device=self.device),
                    "selected_counts": torch.zeros(self.args.num_experts, device=self.device),
                    "overflow_counts": torch.zeros(self.args.num_experts, device=self.device),
                    "avg_router_probs": torch.zeros(self.args.num_experts, device=self.device),
                    "capacity": stats.get("capacity", 0),
                }

            for stat_key in ["expert_activations", "selected_counts", "overflow_counts", "avg_router_probs"]:
                value = stats.get(stat_key)
                if value is not None:
                    total_stats[layer_key][stat_key] += value.to(self.device)
            total_stats[layer_key]["capacity"] = stats.get("capacity", total_stats[layer_key]["capacity"])

    def should_collect_bayes_evidence(self):
        return getattr(self.args, "agg_method", "") == "expert_bayes_meta"

    def get_active_expert_refs(self, layer_stats):
        active_experts = []
        for layer_id, stats in layer_stats.items():
            usage = stats.get("expert_activations")
            if usage is None:
                continue

            for expert_id, expert_usage in enumerate(usage.tolist()):
                expert_usage = int(expert_usage)
                if expert_usage < self.bayes_min_expert_tokens:
                    continue
                active_experts.append((str(layer_id), str(expert_id), expert_usage))
        return active_experts

    def resolve_bayes_cache_device(self):
        if self.bayes_cache_device == "cpu":
            return torch.device("cpu")
        if self.bayes_cache_device == "cuda":
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                return torch.device(self.device)
            return torch.device("cpu")
        if (
            str(self.device).startswith("cuda")
            and torch.cuda.is_available()
            and self.bayes_evidence_batches <= 4
        ):
            return torch.device(self.device)
        return torch.device("cpu")

    def move_bayes_cache_tensor(self, tensor, cache_device):
        if cache_device.type == "cuda":
            return tensor.detach().to(device=cache_device).clone()
        return tensor.detach().cpu().clone()

    def estimate_bayes_cache_memory_mb(self, batch_cache_by_expert):
        seen_tensors = {}
        for layer_cache in batch_cache_by_expert.values():
            for expert_cache in layer_cache.values():
                for cache_entry in expert_cache:
                    if len(cache_entry) == 4:
                        _, _, cached_inputs, cached_labels = cache_entry
                    else:
                        _, cached_inputs, cached_labels = cache_entry
                    for tensor in [cached_inputs, cached_labels]:
                        if torch.is_tensor(tensor):
                            seen_tensors[id(tensor)] = tensor

        total_bytes = 0
        for tensor in seen_tensors.values():
            total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes / (1024.0 * 1024.0)

    def get_cached_batch_device(self, batch_cache_by_expert):
        for layer_cache in batch_cache_by_expert.values():
            for expert_cache in layer_cache.values():
                for cache_entry in expert_cache:
                    if len(cache_entry) == 4:
                        return str(cache_entry[2].device)
                    if len(cache_entry) >= 3:
                        return str(cache_entry[1].device)
        return str(self.resolve_bayes_cache_device())

    def update_bayes_batch_cache(self, batch_cache_by_expert, inputs, labels, layer_stats):
        if not self.should_collect_bayes_evidence():
            return

        cache_device = self.resolve_bayes_cache_device()
        cached_inputs = self.move_bayes_cache_tensor(inputs, cache_device)
        cached_labels = self.move_bayes_cache_tensor(labels, cache_device)

        for layer_id, stats in layer_stats.items():
            sample_hits_by_expert = stats.get("sample_hits_by_expert")
            if sample_hits_by_expert is not None:
                sample_hits_by_expert = sample_hits_by_expert.detach().cpu()
                for expert_id, sample_hits in enumerate(sample_hits_by_expert):
                    sample_indices = torch.nonzero(sample_hits > 0, as_tuple=False).flatten()
                    if sample_indices.numel() == 0:
                        continue

                    expert_score = int(sample_hits[sample_indices].sum().item())
                    sample_indices = sample_indices.to(device=cache_device)
                    expert_inputs = cached_inputs.index_select(0, sample_indices).clone()
                    expert_labels = cached_labels.index_select(0, sample_indices).clone()
                    expert_cache = batch_cache_by_expert.setdefault(str(layer_id), {}).setdefault(str(expert_id), [])
                    expert_cache.append((
                        expert_score,
                        int(sample_indices.numel()),
                        expert_inputs,
                        expert_labels,
                    ))
                    expert_cache.sort(key=lambda item: (item[0], item[1]), reverse=True)
                    if len(expert_cache) > self.bayes_evidence_batches:
                        del expert_cache[self.bayes_evidence_batches:]
                continue

            usage = stats.get("expert_activations")
            if usage is None:
                continue
            for expert_id, expert_usage in enumerate(usage.tolist()):
                expert_usage = int(expert_usage)
                if expert_usage <= 0:
                    continue
                expert_cache = batch_cache_by_expert.setdefault(str(layer_id), {}).setdefault(str(expert_id), [])
                expert_cache.append((expert_usage, cached_inputs.clone(), cached_labels.clone()))
                expert_cache.sort(key=lambda item: item[0], reverse=True)
                if len(expert_cache) > self.bayes_evidence_batches:
                    del expert_cache[self.bayes_evidence_batches:]

    def get_expert_batch_cache(self, batch_cache_by_expert, layer_id, expert_id):
        layer_cache = batch_cache_by_expert.get(str(layer_id), {})
        expert_cache = layer_cache.get(str(expert_id), [])
        batch_cache = []
        for cache_entry in expert_cache:
            if len(cache_entry) == 4:
                _, _, cached_inputs, cached_labels = cache_entry
            else:
                _, cached_inputs, cached_labels = cache_entry
            batch_cache.append((cached_inputs, cached_labels))
        return batch_cache

    def build_evidence_model(self):
        evidence_model = build_model_from_args(self.args)
        cpu_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        evidence_model.load_state_dict(cpu_state_dict)
        return evidence_model

    def get_expert_param_prefix(self, layer_id, expert_id):
        return f"blocks.{layer_id}.ffn.experts.{expert_id}."

    def backup_expert_params(self, model, layer_id, expert_id):
        prefix = self.get_expert_param_prefix(layer_id, expert_id)
        backup = {}
        for name, param in model.named_parameters():
            if name.startswith(prefix):
                backup[name] = param.detach().clone()
        return backup

    def restore_expert_params(self, model, backup):
        if not backup:
            return
        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in backup.items():
                param_dict[name].copy_(value)

    def summarize_named_tensor_state(self, state, high_clip=None):
        values = []
        for value in state.values():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                values.append(value.detach().cpu().float().reshape(-1))

        if not values:
            return {
                "numel": 0,
                "mean": None,
                "min": None,
                "max": None,
                "std": None,
                "low_pct": None,
                "high_pct": None,
            }

        vector = torch.cat(values)
        low_threshold = 1.0001e-4
        summary = {
            "numel": int(vector.numel()),
            "mean": round(float(vector.mean().item()), 6),
            "min": round(float(vector.min().item()), 6),
            "max": round(float(vector.max().item()), 6),
            "std": round(float(vector.std(unbiased=False).item()), 6),
            "low_pct": round(float((vector <= low_threshold).float().mean().item()), 6),
        }
        if high_clip is None or high_clip <= 0:
            summary["high_pct"] = None
        else:
            high_threshold = 0.9999 * float(high_clip)
            summary["high_pct"] = round(float((vector >= high_threshold).float().mean().item()), 6)
        return summary

    def count_cached_samples(self, batch_cache):
        return int(sum(labels.size(0) for _, labels in batch_cache))

    def fit_local_expert_evidence(self, evidence_model, layer_id, expert_id, usage, batch_cache):
        expert_backup = self.backup_expert_params(evidence_model, layer_id, expert_id)
        try:
            mean_state, precision_state, sgld_diag = run_expert_sgld_fit(
                model=evidence_model,
                batch_cache=batch_cache,
                criterion=self.criterion,
                layer_id=layer_id,
                expert_id=expert_id,
                device=self.device,
                steps=self.bayes_sgld_steps,
                burnin=self.bayes_sgld_burnin,
                alp=self.bayes_sgld_lr,
                ai_max=self.bayes_ai_max,
                var_floor=self.bayes_sgld_var_floor,
                precision_mode=self.bayes_precision_mode,
                precision_temperature=self.bayes_precision_temperature,
                precision_target=self.bayes_precision_target,
                precision_min=self.bayes_precision_min,
                precision_max=self.bayes_precision_max,
                precision_eps=self.bayes_precision_eps,
                sgld_concat_cache=self.bayes_sgld_concat_cache,
            )
        finally:
            self.restore_expert_params(evidence_model, expert_backup)
            del expert_backup

        return {
            "usage": usage,
            "num_batches": len(batch_cache),
            "mean_state": mean_state,
            "precision_state": precision_state,
            "sgld_diag": sgld_diag,
        }

    def extract_bayesian_evidence(self, layer_stats, batch_cache_by_expert):
        if not self.should_collect_bayes_evidence():
            return {}

        active_experts = self.get_active_expert_refs(layer_stats)
        cached_expert_count = sum(
            1
            for layer_cache in batch_cache_by_expert.values()
            for expert_cache in layer_cache.values()
            if len(expert_cache) > 0
        )
        self.logger.info(
            f"--client: {self.client_id} --bayes_active_experts : {len(active_experts)} "
            f"--bayes_cached_experts : {cached_expert_count}"
        )
        self.logger.info(
            f"--client: {self.client_id} --bayes_cache_diag "
            f"--bayes_cache_device:{self.bayes_cache_device} "
            f"--resolved_cache_device:{self.resolve_bayes_cache_device()} "
            f"--cached_batch_device:{self.get_cached_batch_device(batch_cache_by_expert)} "
            f"--cached_experts:{cached_expert_count} "
            f"--estimated_cache_memory_mb:{self.estimate_bayes_cache_memory_mb(batch_cache_by_expert):.4f}"
        )
        evidence_by_layer = {}
        build_model_sec = 0.0
        sgld_times = []
        total_start_time = time.perf_counter()
        evidence_model = None
        try:
            if active_experts:
                build_start_time = time.perf_counter()
                evidence_model = self.build_evidence_model()
                evidence_model.to(self.device)
                build_model_sec = time.perf_counter() - build_start_time

            for layer_id, expert_id, usage in active_experts:
                batch_cache = self.get_expert_batch_cache(
                    batch_cache_by_expert=batch_cache_by_expert,
                    layer_id=layer_id,
                    expert_id=expert_id,
                )
                if len(batch_cache) == 0:
                    continue
                layer_evidence = evidence_by_layer.setdefault(layer_id, {})
                sgld_start_time = time.perf_counter()
                expert_evidence = self.fit_local_expert_evidence(
                    evidence_model=evidence_model,
                    layer_id=layer_id,
                    expert_id=expert_id,
                    usage=usage,
                    batch_cache=batch_cache,
                )
                sgld_elapsed = time.perf_counter() - sgld_start_time
                sgld_diag = expert_evidence.get("sgld_diag", {})
                sgld_time_value = sgld_diag.get("sgld_fit_time_sec")
                if isinstance(sgld_time_value, (int, float)):
                    sgld_times.append(float(sgld_time_value))
                else:
                    sgld_times.append(sgld_elapsed)
                layer_evidence[expert_id] = expert_evidence
                if self.bayes_evidence_log_detail:
                    precision_summary = self.summarize_named_tensor_state(
                        expert_evidence["precision_state"],
                        high_clip=self.bayes_ai_max,
                    )
                    mean_summary = self.summarize_named_tensor_state(expert_evidence["mean_state"])
                    self.logger.info(
                        f"--client: {self.client_id} --bayes_evidence_diag "
                        f"--detail:true "
                        f"--layer:{layer_id} --expert:{expert_id} --usage:{int(usage)} "
                        f"--batches:{len(batch_cache)} --cached_samples:{self.count_cached_samples(batch_cache)} "
                        f"--mean_numel:{mean_summary['numel']} "
                        f"--precision_mean:{precision_summary['mean']} "
                        f"--precision_min:{precision_summary['min']} "
                        f"--precision_max:{precision_summary['max']} "
                        f"--precision_std:{precision_summary['std']} "
                        f"--precision_low_pct:{precision_summary['low_pct']} "
                        f"--precision_high_pct:{precision_summary['high_pct']} "
                        f"--local_precision_mean:{precision_summary['mean']} "
                        f"--local_precision_min:{precision_summary['min']} "
                        f"--local_precision_max:{precision_summary['max']} "
                        f"--local_precision_std:{precision_summary['std']} "
                        f"--sgld_samples:{sgld_diag.get('sample_count')} "
                        f"--sgld_lr:{sgld_diag.get('sgld_lr')} "
                        f"--sgld_var_floor:{sgld_diag.get('sgld_var_floor')} "
                        f"--precision_mode:{sgld_diag.get('precision_mode')} "
                        f"--precision_temperature:{sgld_diag.get('precision_temperature')} "
                        f"--precision_target:{sgld_diag.get('precision_target')} "
                        f"--raw_var_mean:{sgld_diag.get('raw_var_mean')} "
                        f"--raw_var_min:{sgld_diag.get('raw_var_min')} "
                        f"--raw_var_max:{sgld_diag.get('raw_var_max')} "
                        f"--raw_var_under_floor_pct:{sgld_diag.get('raw_var_under_floor_pct')} "
                        f"--unclipped_precision_mean:{sgld_diag.get('unclipped_precision_mean')} "
                        f"--unclipped_precision_max:{sgld_diag.get('unclipped_precision_max')} "
                        f"--unclipped_precision_over_ai_max_pct:"
                        f"{sgld_diag.get('unclipped_precision_over_ai_max_pct')}"
                    )
        finally:
            if evidence_model is not None:
                del evidence_model
            if (
                str(self.device).startswith("cuda")
                and torch.cuda.is_available()
                and self.bayes_empty_cache_after_client_evidence
            ):
                torch.cuda.empty_cache()

        total_evidence_sec = time.perf_counter() - total_start_time
        per_expert_mean_sec = sum(sgld_times) / max(len(sgld_times), 1)
        per_expert_max_sec = max(sgld_times) if sgld_times else 0.0
        per_expert_min_sec = min(sgld_times) if sgld_times else 0.0
        self.logger.info(
            f"--client: {self.client_id} --bayes_evidence_time "
            f"--bayes_total_evidence_time_sec:{total_evidence_sec:.4f} "
            f"--bayes_build_model_sec:{build_model_sec:.4f} "
            f"--bayes_per_expert_mean_sec:{per_expert_mean_sec:.4f} "
            f"--bayes_per_expert_max_sec:{per_expert_max_sec:.4f} "
            f"--bayes_per_expert_min_sec:{per_expert_min_sec:.4f} "
            f"--bayes_active_experts:{len(active_experts)} "
            f"--bayes_cached_experts:{cached_expert_count} "
            f"--bayes_evidence_log_detail:{self.bayes_evidence_log_detail} "
            f"--build_model_sec:{build_model_sec:.4f} "
            f"--total_sec:{total_evidence_sec:.4f} "
            f"--per_expert_mean_sec:{per_expert_mean_sec:.4f} "
            f"--active_experts:{len(active_experts)} "
            f"--cached_experts:{cached_expert_count}"
        )

        return evidence_by_layer

    def train(self):
        # 本地训练保持普通监督学习；不同模型通过 forward 返回的 aux loss / stats 接入路由约束和日志。
        self.renew_model()
        self.logger.info(
            f"--client: {self.client_id} --round:{self.c_T + 1} "
            f"--learning_rate:{self.current_learning_rate:.8f} "
            f"--lr_scheduler:{getattr(self.args, 'lr_scheduler', 'none')}"
        )

        local_train_start = time.perf_counter()
        last_avg_router_probs = torch.zeros(self.args.num_experts, device=self.device)
        local_usage_total = torch.zeros(self.args.num_experts, device=self.device)
        local_layer_usage_total = {}
        bayes_batch_cache_by_expert = {}

        for epoch in range(self.client_epochs):
            self.model.train()
            running_loss = 0.0
            running_aux_loss = 0.0
            running_z_loss = 0.0
            running_corrects = 0
            total_samples = 0
            usage_total = torch.zeros(self.args.num_experts, device=self.device)
            layer_usage_total = {}
            router_prob_sum = torch.zeros(self.args.num_experts, device=self.device)

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                result = self.model(inputs)
                outputs = result["logits"]
                extra_loss, router_aux_loss, router_z_loss = self.get_auxiliary_losses(result)
                loss = self.criterion(outputs, labels) + extra_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_aux_loss += router_aux_loss.item() * batch_size
                running_z_loss += router_z_loss.item() * batch_size
                total_samples += batch_size
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

                usage_total += self.get_expert_activations(result)
                batch_layer_stats = self.get_layer_expert_stats(result)
                self.add_layer_stats(layer_usage_total, batch_layer_stats)
                self.update_bayes_batch_cache(
                    batch_cache_by_expert=bayes_batch_cache_by_expert,
                    inputs=inputs,
                    labels=labels,
                    layer_stats=batch_layer_stats,
                )
                router_prob_sum += self.get_avg_router_probs(result) * batch_size

            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = running_corrects.double() / len(self.train_loader.dataset)
            avg_aux_loss = running_aux_loss / max(total_samples, 1)
            avg_z_loss = running_z_loss / max(total_samples, 1)
            local_usage_total += usage_total.detach()
            self.add_layer_stats(local_layer_usage_total, layer_usage_total)
            last_avg_router_probs = router_prob_sum / max(total_samples, 1)

            usage_list = [int(v) for v in usage_total.detach().cpu().tolist()]
            router_prob_list = [round(float(v), 4) for v in last_avg_router_probs.detach().cpu().tolist()]
            self.logger.info(
                f"--client: {self.client_id} --epoch:{epoch+1}/{self.client_epochs} "
                f"--train_loss :{train_loss:.4f} --train_acc :{train_acc:.4f} "
                f"--router_aux_loss : {avg_aux_loss:.4f} "
                f"--router_z_loss : {avg_z_loss:.4f} "
                f"--expert_usage : {usage_list} --avg_router_probs : {router_prob_list}"
            )
            if layer_usage_total:
                layer_usage_log = {
                    layer_id: {
                        "expert_activations": [int(v) for v in stats["expert_activations"].detach().cpu().tolist()],
                        "overflow_counts": [int(v) for v in stats["overflow_counts"].detach().cpu().tolist()],
                        "capacity": int(stats["capacity"]),
                    }
                    for layer_id, stats in layer_usage_total.items()
                }
                self.logger.info(f"--client: {self.client_id} --layer_expert_stats : {layer_usage_log}")

            record_dic = {
                'T': self.c_T,
                'client_epoch': epoch+1,
                'client_id': self.client_id,
                "train_loss": train_loss,
                "train_acc": train_acc.item(),
                "router_aux_loss": avg_aux_loss,
                "router_z_loss": avg_z_loss,
            }
            record_result(record_dic=record_dic, args=self.args)

        local_train_time = time.perf_counter() - local_train_start
        local_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        if self.save_client_models:
            self.save_client_model(local_state_dict)
        layer_stats_cpu = {
            layer_id: {
                stat_key: (value.detach().cpu() if torch.is_tensor(value) else value)
                for stat_key, value in stats.items()
            }
            for layer_id, stats in local_layer_usage_total.items()
        }
        bayes_evidence_start = time.perf_counter()
        bayes_evidence = self.extract_bayesian_evidence(layer_stats_cpu, bayes_batch_cache_by_expert)
        bayes_evidence_time = time.perf_counter() - bayes_evidence_start
        return {
            "local_state_dict": local_state_dict,
            "expert_activations": local_usage_total.detach().cpu(),
            "expert_stats_by_layer": layer_stats_cpu,
            "expert_activations_by_layer": {
                layer_id: stats["expert_activations"]
                for layer_id, stats in layer_stats_cpu.items()
            },
            "bayes_evidence_by_layer": bayes_evidence,
            "local_train_time": local_train_time,
            "bayes_evidence_time": bayes_evidence_time,
        }
