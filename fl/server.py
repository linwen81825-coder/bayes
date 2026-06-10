import json
import os
import time
from types import SimpleNamespace

import torch
from torch import nn
from tqdm import tqdm

from data.loader import (
    build_client_train_loader,
    build_global_eval_loader,
    get_client_train_size,
    load_partition_meta,
)
from fl.aggregators import build_aggregator
from fl.client import Client
from model import build_model_from_args
from utils.utils import (
    capture_rng_state,
    init_result_csv,
    init_server_result_csv,
    record_server_result,
    restore_rng_state,
)


class Server:
    # Server 表示联邦学习中的服务端。
    # 它不直接训练全部数据，而是负责初始化模型、调度客户端训练、聚合客户端模型。
    def __init__(self, args: SimpleNamespace, logger):
        self.args = args
        self.aggregator = build_aggregator(self.args)
        # 基础联邦训练配置。
        self.num_clients = self.args.num_clients
        self.server_epochs = self.args.server_epochs
        # 客户端编号从 1 开始，例如 num_clients=4 时为 [1, 2, 3, 4]。
        self.clientsID_list = [i + 1 for i in range(self.num_clients)]
        self.device = self.args.device
        self.logger = logger
        # 服务端模型保存路径，例如 ./save/model/server.pth。
        self.model_path = os.path.join(self.args.model_save_path, "server.pth")
        self.resume_enabled = bool(getattr(self.args, "resume", False))
        self.start_round = 0
        self.checkpoint_dir = os.path.join(self.args.model_save_path, "checkpoints")

        os.makedirs(self.args.model_save_path, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.partition_meta = load_partition_meta(self.args)
        self.global_test_loader = build_global_eval_loader(
            args=self.args,
            split="global_test",
            meta=self.partition_meta,
        )
        self.cache_client_train_loaders = bool(
            getattr(self.args, "cache_client_train_loaders", False)
        )
        self.client_train_loader_cache = {}
        if self.cache_client_train_loaders:
            # 单进程顺序训练时缓存 DataLoader，可复用 persistent workers。
            self.client_train_loader_cache = {
                client_id: build_client_train_loader(
                    args=self.args,
                    client_id=client_id,
                    meta=self.partition_meta,
                )
                for client_id in self.clientsID_list
            }
        self.logger.info(
            f"--cache_client_train_loaders : {str(self.cache_client_train_loaders).lower()}\n"
        )
        self.logger.info(
            f"--num_cached_client_loaders : {len(self.client_train_loader_cache)}\n"
        )
        self.num_experts = self.args.num_experts
        self.criterion = nn.CrossEntropyLoss()
        # UOC-FOGA 诊断 A/C 的 fixed balanced reference set。
        # 只做日志诊断，不参与训练、PISM loss 或聚合权重。
        self.uoc_foga_ref_grad_diag_enabled = bool(
            getattr(self.args, "uoc_foga_ref_grad_diag_enabled", False)
        )
        self.uoc_foga_ref_step_diag_enabled = bool(
            getattr(self.args, "uoc_foga_ref_step_diag_enabled", False)
        )
        self.uoc_foga_ref_grad_diag_every = max(
            1, int(getattr(self.args, "uoc_foga_ref_grad_diag_every", 1))
        )
        self.uoc_foga_ref_step_diag_every = max(
            1, int(getattr(self.args, "uoc_foga_ref_step_diag_every", 1))
        )
        self.uoc_foga_ref_query_per_class = max(
            1, int(getattr(self.args, "uoc_foga_ref_query_per_class", 8))
        )
        self.reference_inputs = None
        self.reference_labels = None
        if self.uoc_foga_ref_grad_diag_enabled or self.uoc_foga_ref_step_diag_enabled:
            self._build_fixed_balanced_reference_batch()

        if self.resume_enabled:
            self.init_resume_training_state()
            init_result_csv(self.args, overwrite=False)
            init_server_result_csv(self.args, overwrite=False)
        else:
            self.clear_old_checkpoints()
            self.init_fresh_training_state()
            init_result_csv(self.args, overwrite=True)
            init_server_result_csv(self.args, overwrite=True)

    def init_fresh_training_state(self):
        """从头训练：初始化随机全局模型并覆盖旧结果。"""
        self.model = build_model_from_args(self.args)
        self.best_test_acc = -1.0
        self.best_test_loss = float("inf")
        self.best_round = 0
        self.best_state_dict = None
        self.start_round = 0
        self.save_server_model()
        if not bool(getattr(self.args, "in_memory_client_updates", True)):
            self.sync_clients_model()

    def init_resume_training_state(self):
        """断点续训：从 checkpoint 恢复服务端模型和训练状态。"""
        self.model = build_model_from_args(self.args)
        checkpoint_path = self.resolve_resume_checkpoint_path()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.validate_training_checkpoint(checkpoint, checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_test_acc = float(checkpoint["best_test_acc"])
        self.best_test_loss = float(checkpoint["best_test_loss"])
        self.best_round = int(checkpoint["best_round"])
        self.best_state_dict = checkpoint.get("best_state_dict")
        self.start_round = int(checkpoint["round_completed"])

        restore_rng_state(checkpoint.get("rng_state"))

        aggregator_state = checkpoint.get("aggregator_state", None)
        if aggregator_state is not None and hasattr(self.aggregator, "load_checkpoint_state"):
            self.aggregator.load_checkpoint_state(
                aggregator_state,
                map_location=self.device,
            )
            self.logger.info("--aggregator_checkpoint_loaded : true\n")
            pism_steps = getattr(self.aggregator, "pism_update_steps", None)
            if pism_steps is not None:
                self.logger.info(f"--pism_update_steps_loaded : {int(pism_steps)}\n")
        else:
            self.logger.info("--aggregator_checkpoint_loaded : false\n")

        self.save_server_model()
        if not bool(getattr(self.args, "in_memory_client_updates", True)):
            self.sync_clients_model()

        self.logger.info(f"[Resume] Loaded checkpoint from {checkpoint_path}")
        self.logger.info(
            f"[Resume] Completed rounds: {self.start_round}, next round: {self.start_round + 1}"
        )

    def clear_old_checkpoints(self):
        """从头训练时清理旧 checkpoint，避免 latest.pth 残留造成误用。"""
        if not os.path.isdir(self.checkpoint_dir):
            return

        for filename in os.listdir(self.checkpoint_dir):
            if filename == "latest.pth" or (
                filename.startswith("round_") and filename.endswith(".pth")
            ):
                os.remove(os.path.join(self.checkpoint_dir, filename))

    def resolve_resume_checkpoint_path(self):
        """解析 resume_checkpoint 配置。"""
        resume_checkpoint = getattr(self.args, "resume_checkpoint", "latest")
        if resume_checkpoint == "latest":
            checkpoint_path = os.path.join(self.checkpoint_dir, "latest.pth")
        else:
            checkpoint_path = resume_checkpoint

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"resume=true but checkpoint not found: {checkpoint_path}. "
                "If you want to start a fresh run, set resume: false."
            )
        return checkpoint_path

    def validate_training_checkpoint(self, checkpoint, checkpoint_path):
        """检查训练 checkpoint 是否包含断点续训所需字段。"""
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Training checkpoint must be a dict: {checkpoint_path}, "
                f"got {type(checkpoint).__name__}."
            )

        required_keys = {
            "round_completed",
            "model_state_dict",
            "best_test_acc",
            "best_test_loss",
            "best_round",
            "rng_state",
        }
        missing_keys = required_keys - set(checkpoint.keys())
        if missing_keys:
            raise ValueError(
                f"Training checkpoint {checkpoint_path} is missing keys: {sorted(missing_keys)}"
            )

        round_completed = int(checkpoint["round_completed"])
        if round_completed < 0:
            raise ValueError(
                f"Training checkpoint {checkpoint_path} has negative round_completed: {round_completed}"
            )
        if round_completed > self.server_epochs:
            raise ValueError(
                f"Training checkpoint round_completed={round_completed} exceeds "
                f"server_epochs={self.server_epochs}. Please check the config."
            )

    def save_training_checkpoint(self, round_completed):
        """保存训练 checkpoint，用于后续从下一轮继续训练。"""
        aggregator_state = None
        if hasattr(self.aggregator, "get_checkpoint_state"):
            # 只保存聚合器内部轻量状态；不保存 evidence、client updates 或日志缓存。
            aggregator_state = self.aggregator.get_checkpoint_state()

        checkpoint = {
            "round_completed": int(round_completed),
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            },
            "best_test_acc": float(self.best_test_acc),
            "best_test_loss": float(self.best_test_loss),
            "best_round": int(self.best_round),
            "best_state_dict": (
                None
                if self.best_state_dict is None
                else {
                    key: value.detach().cpu().clone()
                    for key, value in self.best_state_dict.items()
                }
            ),
            "rng_state": capture_rng_state(),
            "config": dict(vars(self.args)),
            "aggregator_state": aggregator_state,
        }

        round_path = os.path.join(
            self.checkpoint_dir,
            f"round_{round_completed:04d}.pth",
        )
        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")

        torch.save(checkpoint, round_path)
        torch.save(checkpoint, latest_path)

        self.logger.info(f"--checkpoint_saved : {round_path}\n")
        self.logger.info(f"--aggregator_checkpoint_saved : {aggregator_state is not None}\n")
        pism_steps = getattr(self.aggregator, "pism_update_steps", None)
        if pism_steps is not None:
            self.logger.info(f"--pism_update_steps : {int(pism_steps)}\n")

    def save_server_model(self):
        # 保存当前服务端模型参数到 server.pth。
        torch.save(self.model.state_dict(), self.model_path)

    def get_cpu_state_dict(self):
        """获取当前服务端模型的 CPU state_dict，用于分发给客户端。"""
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def sync_clients_model(self):
        server_state_dict = self.get_cpu_state_dict()
        for id in self.clientsID_list:
            # pth fallback 模式下，客户端文件保存同一个服务端 state_dict。
            model_path = os.path.join(self.args.model_save_path, f"{id}.pth")
            torch.save(server_state_dict, model_path)

    def _summarize_uoc_foga_stats(self, uoc_foga_stats):
        if not isinstance(uoc_foga_stats, dict):
            return None

        expert_metrics = [
            metric
            for layer_stats in uoc_foga_stats.values()
            if isinstance(layer_stats, dict)
            for metric in layer_stats.values()
            if isinstance(metric, dict)
        ]
        total_experts = len(expert_metrics)
        fallback_reason_counts = {}
        updated_experts = 0
        score_means = []
        score_metrics = []
        weight_entropies = []
        query_modes = []
        query_fallback_to_random_count = 0
        query_token_ratio_means = []
        query_entropy_means = []
        query_pool_after_ratio_filter_sizes = []
        grad_dot_valid_score_counts = []
        grad_dot_positive_count_estimates = []
        grad_dot_score_items = []
        grad_dot_score_min_values = []
        grad_dot_score_max_values = []
        grad_dot_client_set_size_items = []
        grad_dot_client_set_size_min_values = []
        grad_dot_client_set_size_max_values = []
        cos_delta_neg_gclient_items = []
        cos_delta_neg_gclient_positive_count_estimates = []
        delta_consensus_valid_score_counts = []
        delta_consensus_positive_count_estimates = []
        delta_consensus_score_items = []
        delta_consensus_score_min_values = []
        delta_consensus_score_max_values = []
        delta_consensus_ref_client_items = []

        for metric in expert_metrics:
            fallback_reason = metric.get("fallback_reason")
            reason = fallback_reason if fallback_reason is not None else "none"
            fallback_reason_counts[str(reason)] = fallback_reason_counts.get(str(reason), 0) + 1
            if fallback_reason is None:
                updated_experts += 1
            score_mean = metric.get("score_mean")
            if score_mean is not None:
                score_means.append(float(score_mean))
            score_metric = metric.get("score_metric")
            if score_metric is not None:
                score_metrics.append(str(score_metric))
            weight_entropy = metric.get("weight_entropy")
            if weight_entropy is not None:
                weight_entropies.append(float(weight_entropy))

            query_mode = metric.get("query_select_mode")
            if query_mode is not None:
                query_modes.append(str(query_mode))
            if bool(metric.get("fallback_to_random_used", False)):
                query_fallback_to_random_count += 1
            token_ratio_mean = metric.get("expert_token_ratio_mean")
            if token_ratio_mean is not None:
                query_token_ratio_means.append(float(token_ratio_mean))
            query_entropy_mean = metric.get("query_entropy_mean")
            if query_entropy_mean is not None:
                query_entropy_means.append(float(query_entropy_mean))
            pool_after_ratio_filter_size = metric.get("pool_size_after_token_ratio_filter")
            if pool_after_ratio_filter_size is not None:
                query_pool_after_ratio_filter_sizes.append(float(pool_after_ratio_filter_size))

            grad_dot_valid_count = metric.get("grad_dot_valid_scores")
            if grad_dot_valid_count is not None:
                grad_dot_valid_count = int(grad_dot_valid_count)
                grad_dot_valid_score_counts.append(grad_dot_valid_count)
                grad_dot_positive_frac = metric.get("grad_dot_positive_frac")
                if grad_dot_positive_frac is not None and grad_dot_valid_count > 0:
                    grad_dot_positive_count_estimates.append(
                        float(grad_dot_positive_frac) * grad_dot_valid_count
                    )
                grad_dot_score_mean = metric.get("grad_dot_score_mean")
                if grad_dot_score_mean is not None and grad_dot_valid_count > 0:
                    grad_dot_score_items.append((
                        float(grad_dot_score_mean),
                        float(metric.get("grad_dot_score_std") or 0.0),
                        grad_dot_valid_count,
                    ))
                grad_dot_score_min = metric.get("grad_dot_score_min")
                if grad_dot_score_min is not None:
                    grad_dot_score_min_values.append(float(grad_dot_score_min))
                grad_dot_score_max = metric.get("grad_dot_score_max")
                if grad_dot_score_max is not None:
                    grad_dot_score_max_values.append(float(grad_dot_score_max))

                client_set_size_mean = metric.get("grad_dot_client_set_size_mean")
                if client_set_size_mean is not None and grad_dot_valid_count > 0:
                    grad_dot_client_set_size_items.append((
                        float(client_set_size_mean),
                        grad_dot_valid_count,
                    ))
                client_set_size_min = metric.get("grad_dot_client_set_size_min")
                if client_set_size_min is not None:
                    grad_dot_client_set_size_min_values.append(float(client_set_size_min))
                client_set_size_max = metric.get("grad_dot_client_set_size_max")
                if client_set_size_max is not None:
                    grad_dot_client_set_size_max_values.append(float(client_set_size_max))

                cos_mean = metric.get("cos_delta_neg_gclient_mean")
                if cos_mean is not None and grad_dot_valid_count > 0:
                    cos_delta_neg_gclient_items.append((float(cos_mean), grad_dot_valid_count))
                cos_positive_frac = metric.get("cos_delta_neg_gclient_positive_frac")
                if cos_positive_frac is not None and grad_dot_valid_count > 0:
                    cos_delta_neg_gclient_positive_count_estimates.append(
                        float(cos_positive_frac) * grad_dot_valid_count
                    )

            delta_consensus_valid_count = metric.get("delta_consensus_valid_scores")
            if delta_consensus_valid_count is not None:
                delta_consensus_valid_count = int(delta_consensus_valid_count)
                delta_consensus_valid_score_counts.append(delta_consensus_valid_count)
                delta_positive_frac = metric.get("delta_consensus_positive_frac")
                if delta_positive_frac is not None and delta_consensus_valid_count > 0:
                    delta_consensus_positive_count_estimates.append(
                        float(delta_positive_frac) * delta_consensus_valid_count
                    )
                delta_score_mean = metric.get("delta_consensus_score_mean")
                if delta_score_mean is not None and delta_consensus_valid_count > 0:
                    delta_consensus_score_items.append((
                        float(delta_score_mean),
                        float(metric.get("delta_consensus_score_std") or 0.0),
                        delta_consensus_valid_count,
                    ))
                delta_score_min = metric.get("delta_consensus_score_min")
                if delta_score_min is not None:
                    delta_consensus_score_min_values.append(float(delta_score_min))
                delta_score_max = metric.get("delta_consensus_score_max")
                if delta_score_max is not None:
                    delta_consensus_score_max_values.append(float(delta_score_max))
                ref_clients_mean = metric.get("delta_consensus_ref_clients_mean")
                if ref_clients_mean is not None and delta_consensus_valid_count > 0:
                    delta_consensus_ref_client_items.append((
                        float(ref_clients_mean),
                        delta_consensus_valid_count,
                    ))

        query_select_mode = (
            query_modes[0]
            if query_modes
            else getattr(self.args, "uoc_foga_query_select_mode", "class_balanced_random")
        )
        score_metric = (
            score_metrics[0]
            if score_metrics
            else getattr(self.args, "uoc_foga_score_metric", "cosine")
        )
        total_grad_dot_valid_scores = sum(grad_dot_valid_score_counts)
        grad_dot_score_mean = None
        grad_dot_score_std = None
        if grad_dot_score_items:
            score_weight_total = sum(item[2] for item in grad_dot_score_items)
            if score_weight_total > 0:
                grad_dot_score_mean = sum(
                    mean_value * count
                    for mean_value, _, count in grad_dot_score_items
                ) / score_weight_total
                grad_dot_variance = sum(
                    count * (std_value ** 2 + (mean_value - grad_dot_score_mean) ** 2)
                    for mean_value, std_value, count in grad_dot_score_items
                ) / score_weight_total
                grad_dot_score_std = float(grad_dot_variance ** 0.5)

        grad_dot_client_set_size_mean = None
        if grad_dot_client_set_size_items:
            set_size_weight_total = sum(count for _, count in grad_dot_client_set_size_items)
            if set_size_weight_total > 0:
                grad_dot_client_set_size_mean = sum(
                    mean_value * count
                    for mean_value, count in grad_dot_client_set_size_items
                ) / set_size_weight_total

        cos_delta_neg_gclient_mean = None
        if cos_delta_neg_gclient_items:
            cos_weight_total = sum(count for _, count in cos_delta_neg_gclient_items)
            if cos_weight_total > 0:
                cos_delta_neg_gclient_mean = sum(
                    mean_value * count
                    for mean_value, count in cos_delta_neg_gclient_items
                ) / cos_weight_total

        total_delta_consensus_valid_scores = sum(delta_consensus_valid_score_counts)
        delta_consensus_score_mean = None
        delta_consensus_score_std = None
        if delta_consensus_score_items:
            delta_weight_total = sum(item[2] for item in delta_consensus_score_items)
            if delta_weight_total > 0:
                delta_consensus_score_mean = sum(
                    mean_value * count
                    for mean_value, _, count in delta_consensus_score_items
                ) / delta_weight_total
                delta_consensus_variance = sum(
                    count * (std_value ** 2 + (mean_value - delta_consensus_score_mean) ** 2)
                    for mean_value, std_value, count in delta_consensus_score_items
                ) / delta_weight_total
                delta_consensus_score_std = float(delta_consensus_variance ** 0.5)

        delta_consensus_ref_clients_mean = None
        if delta_consensus_ref_client_items:
            ref_weight_total = sum(count for _, count in delta_consensus_ref_client_items)
            if ref_weight_total > 0:
                delta_consensus_ref_clients_mean = sum(
                    mean_value * count
                    for mean_value, count in delta_consensus_ref_client_items
                ) / ref_weight_total

        return {
            "uoc_foga_updated_experts": updated_experts,
            "uoc_foga_fallback_experts": total_experts - updated_experts,
            "uoc_foga_fallback_reason_counts": fallback_reason_counts,
            "uoc_foga_weight_entropy_mean": (
                sum(weight_entropies) / len(weight_entropies)
                if weight_entropies
                else None
            ),
            "uoc_foga_score_mean_mean": (
                sum(score_means) / len(score_means)
                if score_means
                else None
            ),
            "uoc_foga_score_metric": score_metric,
            "uoc_foga_query_select_mode": query_select_mode,
            "uoc_foga_query_fallback_to_random_count": query_fallback_to_random_count,
            "uoc_foga_query_token_ratio_mean_mean": (
                sum(query_token_ratio_means) / len(query_token_ratio_means)
                if query_token_ratio_means
                else None
            ),
            "uoc_foga_query_entropy_mean_mean": (
                sum(query_entropy_means) / len(query_entropy_means)
                if query_entropy_means
                else None
            ),
            "uoc_foga_query_pool_after_ratio_filter_mean": (
                sum(query_pool_after_ratio_filter_sizes) / len(query_pool_after_ratio_filter_sizes)
                if query_pool_after_ratio_filter_sizes
                else None
            ),
            "grad_dot_valid_scores": total_grad_dot_valid_scores,
            "grad_dot_positive_frac": (
                sum(grad_dot_positive_count_estimates) / total_grad_dot_valid_scores
                if total_grad_dot_valid_scores > 0 and grad_dot_positive_count_estimates
                else None
            ),
            "grad_dot_score_mean": grad_dot_score_mean,
            "grad_dot_score_std": grad_dot_score_std,
            "grad_dot_score_min": min(grad_dot_score_min_values) if grad_dot_score_min_values else None,
            "grad_dot_score_max": max(grad_dot_score_max_values) if grad_dot_score_max_values else None,
            "grad_dot_client_set_size_mean": grad_dot_client_set_size_mean,
            "grad_dot_client_set_size_min": (
                min(grad_dot_client_set_size_min_values)
                if grad_dot_client_set_size_min_values
                else None
            ),
            "grad_dot_client_set_size_max": (
                max(grad_dot_client_set_size_max_values)
                if grad_dot_client_set_size_max_values
                else None
            ),
            "cos_delta_neg_gclient_mean": cos_delta_neg_gclient_mean,
            "cos_delta_neg_gclient_positive_frac": (
                sum(cos_delta_neg_gclient_positive_count_estimates) / total_grad_dot_valid_scores
                if total_grad_dot_valid_scores > 0 and cos_delta_neg_gclient_positive_count_estimates
                else None
            ),
            "delta_consensus_valid_scores": total_delta_consensus_valid_scores,
            "delta_consensus_positive_frac": (
                sum(delta_consensus_positive_count_estimates) / total_delta_consensus_valid_scores
                if total_delta_consensus_valid_scores > 0 and delta_consensus_positive_count_estimates
                else None
            ),
            "delta_consensus_score_mean": delta_consensus_score_mean,
            "delta_consensus_score_std": delta_consensus_score_std,
            "delta_consensus_score_min": (
                min(delta_consensus_score_min_values)
                if delta_consensus_score_min_values
                else None
            ),
            "delta_consensus_score_max": (
                max(delta_consensus_score_max_values)
                if delta_consensus_score_max_values
                else None
            ),
            "delta_consensus_ref_clients_mean": delta_consensus_ref_clients_mean,
        }

    def _infer_num_classes(self):
        num_classes = getattr(self.args, "num_classes", None)
        if num_classes is not None:
            return int(num_classes)
        data_name = str(getattr(self.args, "data_name", "cifar10")).lower()
        if data_name == "cifar100":
            return 100
        return 10

    def _build_fixed_balanced_reference_batch(self):
        """从 global eval loader 中固定抽取 class-balanced reference batch，只做诊断。"""
        num_classes = self._infer_num_classes()
        per_class = self.uoc_foga_ref_query_per_class
        selected_inputs = []
        selected_labels = []
        class_counts = {class_id: 0 for class_id in range(num_classes)}
        target_total = num_classes * per_class

        for inputs, labels in self.global_test_loader:
            labels_cpu = labels.detach().cpu().long()
            inputs_cpu = inputs.detach().cpu()
            for sample_idx in range(labels_cpu.size(0)):
                label = int(labels_cpu[sample_idx].item())
                if label not in class_counts:
                    continue
                if class_counts[label] >= per_class:
                    continue
                selected_inputs.append(inputs_cpu[sample_idx].clone())
                selected_labels.append(labels_cpu[sample_idx].clone())
                class_counts[label] += 1
                if len(selected_labels) >= target_total:
                    break
            if len(selected_labels) >= target_total:
                break

        if not selected_inputs:
            self.logger.info("--uoc_foga_ref_batch_built : false --reason : no_reference_samples\n")
            self.reference_inputs = None
            self.reference_labels = None
            return

        self.reference_inputs = torch.stack(selected_inputs, dim=0)
        self.reference_labels = torch.stack(selected_labels, dim=0).long()
        self.logger.info(
            "--uoc_foga_ref_batch_built : true "
            f"--ref_query_per_class : {per_class} "
            f"--ref_total_samples : {int(self.reference_labels.numel())} "
            f"--ref_class_counts : {[class_counts[idx] for idx in range(num_classes)]}\n"
        )

    def _forward_model_collecting_uoc_evidence(self, inputs):
        """
        收集 fixed reference batch 的 UOC evidence。

        HybridSwitchTransformer 的普通 forward() 只返回 logits / router stats，
        不返回 hidden/residual 这类 UOC evidence。真正用于 UOC 的接口是
        model.collect_uoc_evidence(...)，所以这里必须优先调用它。
        """
        if hasattr(self.model, "collect_uoc_evidence"):
            return {
                "uoc_evidence_by_layer": self.model.collect_uoc_evidence(
                    inputs,
                    max_samples=None,
                    use_top1=bool(getattr(self.args, "uoc_foga_use_top1", True)),
                )
            }

        forward_attempts = (
            {"return_uoc_evidence": True},
            {"collect_uoc_evidence": True},
            {"return_evidence": True},
            {"collect_evidence": True},
            {},
        )
        last_error = None
        for kwargs in forward_attempts:
            try:
                return self.model(inputs, **kwargs)
            except TypeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return self.model(inputs)

    def _is_layer_evidence_dict(self, value):
        """
        判断一个 dict 是否已经是 {layer_id: layer_evidence} 格式。

        collect_uoc_evidence() 可能直接返回：
            {"1": {"hidden": ..., "residual": ...}, "3": ...}

        普通 forward 包装接口可能返回：
            {"uoc_evidence_by_layer": {"1": ...}}
        """
        if not isinstance(value, dict) or not value:
            return False
        for layer_value in value.values():
            if isinstance(layer_value, dict) and torch.is_tensor(layer_value.get("hidden")):
                return True
        return False

    def _extract_uoc_evidence_from_output(self, output):
        if not isinstance(output, dict):
            return None

        # collect_uoc_evidence() 直接返回的 layer evidence。
        if self._is_layer_evidence_dict(output):
            return output

        # forward() 包装返回的 layer evidence。
        for key in (
            "uoc_evidence_by_layer",
            "evidence_by_layer",
            "uoc_evidence",
            "evidence",
            "moe_evidence_by_layer",
        ):
            value = output.get(key)
            if self._is_layer_evidence_dict(value):
                return value

        return None

    def _append_reference_evidence_chunk(self, merged, evidence_by_layer, labels):
        if not isinstance(evidence_by_layer, dict):
            return
        labels_cpu = labels.detach().cpu().long()
        for layer_id, layer_evidence in evidence_by_layer.items():
            if not isinstance(layer_evidence, dict):
                continue
            hidden = layer_evidence.get("hidden")
            if hidden is None or not torch.is_tensor(hidden):
                continue
            sample_count = hidden.size(0)
            if sample_count <= 0:
                continue
            layer_key = str(layer_id)
            target = merged.setdefault(layer_key, {"hidden": [], "labels": [], "residual": []})
            target["hidden"].append(hidden[:sample_count].detach().cpu())

            evidence_labels = layer_evidence.get("labels")
            if torch.is_tensor(evidence_labels) and evidence_labels.dim() > 0:
                target["labels"].append(evidence_labels[:sample_count].detach().cpu().long())
            else:
                target["labels"].append(labels_cpu[:sample_count])

            residual = layer_evidence.get("residual")
            if torch.is_tensor(residual) and residual.dim() > 0 and residual.size(0) >= sample_count:
                target["residual"].append(residual[:sample_count].detach().cpu())

    def _finalize_reference_evidence(self, merged):
        finalized = {}
        for layer_id, chunks in merged.items():
            hidden_chunks = chunks.get("hidden") or []
            label_chunks = chunks.get("labels") or []
            if not hidden_chunks or not label_chunks:
                continue
            hidden = torch.cat(hidden_chunks, dim=0)
            labels = torch.cat(label_chunks, dim=0).long()
            sample_count = min(hidden.size(0), labels.size(0))
            if sample_count <= 0:
                continue
            layer_result = {
                "hidden": hidden[:sample_count],
                "labels": labels[:sample_count],
            }
            residual_chunks = chunks.get("residual") or []
            if len(residual_chunks) == len(hidden_chunks):
                residual = torch.cat(residual_chunks, dim=0)
                if residual.size(0) >= sample_count and residual.shape[1:] == hidden.shape[1:]:
                    layer_result["residual"] = residual[:sample_count]
            finalized[str(layer_id)] = layer_result
        return finalized

    def _collect_reference_uoc_evidence(self, round_completed):
        """每轮聚合前收集 fixed balanced reference 的 hidden/residual/labels。"""
        need_grad_diag = (
            self.uoc_foga_ref_grad_diag_enabled
            and round_completed % self.uoc_foga_ref_grad_diag_every == 0
        )
        need_step_diag = (
            self.uoc_foga_ref_step_diag_enabled
            and round_completed % self.uoc_foga_ref_step_diag_every == 0
        )
        if not (need_grad_diag or need_step_diag):
            return None
        if self.reference_inputs is None or self.reference_labels is None:
            return None

        was_training = self.model.training
        merged = {}
        try:
            self.model.to(self.device)
            self.model.eval()
            inputs = self.reference_inputs.to(self.device)
            labels = self.reference_labels.to(self.device)
            with torch.no_grad():
                output = self._forward_model_collecting_uoc_evidence(inputs)
            evidence_by_layer = self._extract_uoc_evidence_from_output(output)
            self._append_reference_evidence_chunk(merged, evidence_by_layer, labels)
        except Exception as exc:  # 诊断失败不能影响训练。
            self.logger.info(f"--uoc_foga_ref_evidence_failed : {type(exc).__name__}: {exc}\n")
            return None
        finally:
            self.model.train(was_training)

        reference_evidence = self._finalize_reference_evidence(merged)
        self.logger.info(
            "--uoc_foga_ref_evidence_layers : "
            f"{ {layer: int(value['hidden'].shape[0]) for layer, value in reference_evidence.items()} }\n"
        )
        return reference_evidence if reference_evidence else None

    def train(self):
        if self.start_round >= self.server_epochs:
            self.logger.info(
                f"[Resume] Checkpoint already reached server_epochs={self.server_epochs}. Nothing to train."
            )
            return

        total_client_steps = self.server_epochs * len(self.clientsID_list)
        initial_client_steps = self.start_round * len(self.clientsID_list)

    progress_bar = tqdm(
        total=total_client_steps,
        initial=initial_client_steps,
        desc="Experiment progress",
        unit="client",
        dynamic_ncols=True,
        leave=True,
        ascii=False,
        mininterval=0.1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        # 控制台瘦身：默认关闭 tqdm，避免进度条刷屏。
        # 不影响日志文件，因为 tqdm 本来也不写入日志文件。
        disable=bool(getattr(self.args, "quiet_console", True)),
    )
        try:
            # 外层循环是一轮轮服务端通信，也就是联邦学习中的 global round。
            for c_T in range(self.start_round, self.server_epochs):
                round_start_time = time.perf_counter()
                self.logger.info(f"============================== T:{c_T+1} start !!! ===============================\n")
                use_in_memory_updates = bool(getattr(self.args, "in_memory_client_updates", True))
                if use_in_memory_updates:
                    server_state_dict = self.get_cpu_state_dict()
                else:
                    server_state_dict = None
                    self.save_server_model()
                    self.sync_clients_model()

                round_expert_usage_summary = torch.zeros(self.args.num_experts)
                round_layer_stats = {}
                round_client_expert_usages = []
                round_client_uoc_evidences = []
                round_client_stats = []
                client_states = []
                client_train_start_time = time.perf_counter()
                for id in self.clientsID_list:
                    train_loader = (
                        self.client_train_loader_cache.get(id)
                        if self.cache_client_train_loaders
                        else None
                    )
                    # 每个客户端执行本地训练，并返回本轮信息。
                    client_stats = Client(
                        args=self.args,
                        client_id=id,
                        logger=self.logger,
                        c_T=c_T,
                        partition_meta=self.partition_meta,
                        initial_state_dict=server_state_dict if use_in_memory_updates else None,
                        save_model_to_disk=not use_in_memory_updates,
                        train_loader=train_loader,
                    ).train()
                    progress_bar.update(1)
                    progress_bar.set_postfix_str(
                        f"round={c_T + 1}/{self.server_epochs}, client={id}"
                    )
                    progress_bar.refresh()

                    if use_in_memory_updates:
                        if "model_state_dict" not in client_stats:
                            raise KeyError(
                                "in_memory_client_updates=True requires Client.train() to return model_state_dict"
                            )
                        client_states.append(client_stats.pop("model_state_dict"))

                    uoc_evidence = client_stats.get("uoc_evidence_by_layer", {})
                    round_client_uoc_evidences.append(uoc_evidence)
                    # 传给聚合器的 client stats 保持轻量，不重复保存 model_state_dict 或 evidence 大 tensor。
                    round_client_stats.append({
                        "client_id": id,
                        "client_loss": client_stats.get(
                            "client_loss",
                            client_stats.get("train_loss", 0.0),
                        ),
                        "train_loss": client_stats.get("train_loss", None),
                        "train_acc": client_stats.get("train_acc", None),
                        "expert_activations": client_stats.get("expert_activations", None),
                        "expert_stats_by_layer": client_stats.get("expert_stats_by_layer", None),
                        "expert_activations_by_layer": client_stats.get(
                            "expert_activations_by_layer",
                            None,
                        ),
                        "expert_loss_by_layer": client_stats.get(
                            "expert_loss_by_layer",
                            None,
                        ),
                    })

                    client_expert_usage = client_stats["expert_activations"].float().cpu()
                    round_client_expert_usages.append(client_stats)
                    round_expert_usage_summary += client_expert_usage
                    for layer_id, stats in client_stats.get("expert_stats_by_layer", {}).items():
                        if layer_id not in round_layer_stats:
                            round_layer_stats[layer_id] = {
                                "expert_activations": torch.zeros(self.args.num_experts),
                                "overflow_counts": torch.zeros(self.args.num_experts),
                                "capacity": stats.get("capacity", 0),
                            }
                        round_layer_stats[layer_id]["expert_activations"] += stats["expert_activations"].float().cpu()
                        round_layer_stats[layer_id]["overflow_counts"] += stats["overflow_counts"].float().cpu()
                        round_layer_stats[layer_id]["capacity"] = stats.get("capacity", round_layer_stats[layer_id]["capacity"])

                round_client_train_seconds = time.perf_counter() - client_train_start_time
                usage_list = [int(v) for v in round_expert_usage_summary.tolist()]
                self.logger.info(f"--round_expert_usage_summary : {usage_list}\n")
                client_usage_list = [
                    [int(v) for v in stats["expert_activations"].tolist()]
                    for stats in round_client_expert_usages
                ]
                layer_stats_log = {
                    layer_id: {
                        "expert_activations": [int(v) for v in stats["expert_activations"].tolist()],
                        "overflow_counts": [int(v) for v in stats["overflow_counts"].tolist()],
                        "capacity": int(stats["capacity"]),
                    }
                    for layer_id, stats in round_layer_stats.items()
                }
                client_uoc_evidence_counts = []
                for uoc_evidence in round_client_uoc_evidences:
                    if not uoc_evidence:
                        client_uoc_evidence_counts.append(0)
                        continue
                    client_uoc_evidence_counts.append({
                        str(layer_id): int(layer_evidence["hidden"].shape[0])
                        for layer_id, layer_evidence in uoc_evidence.items()
                        if isinstance(layer_evidence, dict) and "hidden" in layer_evidence
                    })
                self.logger.info(f"--client_expert_usage_summary : {client_usage_list}\n")
                self.logger.info(f"--round_expert_stats_by_layer : {layer_stats_log}\n")
                self.logger.info(f"--client_uoc_evidence_counts : {client_uoc_evidence_counts}\n")

                # 所有客户端本地训练完成后，服务端通过聚合器更新全局模型。
                round_completed = c_T + 1
                aggregation_start_time = time.perf_counter()
                reference_uoc_evidence = self._collect_reference_uoc_evidence(round_completed)
                self.aggregation(
                    client_states=client_states if use_in_memory_updates else None,
                    uoc_evidence=round_client_uoc_evidences,
                    reference_uoc_evidence=reference_uoc_evidence,
                    client_stats=round_client_stats,
                    round_index=round_completed,
                )
                round_aggregation_seconds = time.perf_counter() - aggregation_start_time

                eval_every = max(1, int(getattr(self.args, "eval_every", 1)))
                should_eval = round_completed % eval_every == 0 or round_completed >= self.server_epochs
                round_eval_seconds = None
                if should_eval:
                    eval_start_time = time.perf_counter()
                test_loss, test_acc = self.evaluate_global_model(self.global_test_loader)

                round_eval_seconds = time.perf_counter() - eval_start_time
                self.logger.info(f"--server_global_test_loss : {test_loss:.4f} --server_global_test_acc : {test_acc:.4f}\n")

                is_best = self.update_best_model(test_acc=test_acc, test_loss=test_loss, round_id=round_completed)

                # 控制台专用精简摘要：
                # 只有这条日志会通过 train.py 里的 ConsoleSummaryFilter 显示到控制台。
                # 其他完整诊断日志仍然全部写入 logs/*.log。
                self.logger.info(
                    f"round={round_completed:04d}/{self.server_epochs:04d} "
                    f"loss={test_loss:.4f} "
                    f"final_acc={test_acc * 100.0:.2f}% "
                    f"best_acc={self.best_test_acc * 100.0:.2f}% "
                    f"best_round={self.best_round}",
                    extra={"to_console": True},
                )
                    record_server_result(
                        {
                            "phase": "test",
                            "round": round_completed,
                            "test_loss": test_loss,
                            "test_acc": test_acc,
                            "best_test_acc": self.best_test_acc,
                            "best_test_loss": self.best_test_loss,
                            "is_best": int(is_best),
                        },
                        self.args,
                    )
                else:
                    self.logger.info(f"--server_global_test_skipped : true --eval_every : {eval_every}\n")
                    if self.args.expert_agg_method in {"uoc_foga_expert_align", "uoc_foga_pism_expert_align"}:
                        self.model.to("cpu")

                save_server_each_round = bool(getattr(self.args, "save_server_model_each_round", False))
                if (not use_in_memory_updates) or save_server_each_round:
                    self.save_server_model()

                round_checkpoint_seconds = None
                checkpoint_every = max(1, int(getattr(self.args, "checkpoint_every", 1)))
                should_save_checkpoint = (
                    checkpoint_every <= 1
                    or round_completed % checkpoint_every == 0
                    or round_completed >= self.server_epochs
                )
                if should_save_checkpoint:
                    checkpoint_start_time = time.perf_counter()
                    self.save_training_checkpoint(round_completed)
                    round_checkpoint_seconds = time.perf_counter() - checkpoint_start_time
                else:
                    self.logger.info(f"--checkpoint_skipped : true --checkpoint_every : {checkpoint_every}\n")

                round_total_seconds = time.perf_counter() - round_start_time
                self.logger.info(f"--round_client_train_seconds : {round_client_train_seconds:.4f}\n")
                self.logger.info(f"--round_aggregation_seconds : {round_aggregation_seconds:.4f}\n")
                if round_eval_seconds is None:
                    self.logger.info("--round_eval_seconds : None\n")
                else:
                    self.logger.info(f"--round_eval_seconds : {round_eval_seconds:.4f}\n")
                if round_checkpoint_seconds is None:
                    self.logger.info("--round_checkpoint_seconds : None\n")
                else:
                    self.logger.info(f"--round_checkpoint_seconds : {round_checkpoint_seconds:.4f}\n")
                self.logger.info(f"--round_total_seconds : {round_total_seconds:.4f}\n")
        finally:
            progress_bar.close()

        self.logger.info(
            f"--best_global_test_acc : {self.best_test_acc:.4f} "
            f"--best_global_test_loss : {self.best_test_loss:.4f} "
            f"--best_round : {self.best_round}\n"
        )

    def evaluate_global_model(self, data_loader):
        self.model.to(self.device)
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        non_blocking = (
            str(self.device).startswith("cuda")
            and bool(getattr(self.args, "pin_memory", False))
        )

        inference_context = getattr(torch, "inference_mode", torch.no_grad)
        with inference_context():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device, non_blocking=non_blocking)
                labels = labels.to(self.device, non_blocking=non_blocking)
                result = self.model(inputs)
                outputs = result["logits"]
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        average_loss = running_loss / len(data_loader.dataset)
        accuracy = running_corrects.double() / len(data_loader.dataset)
        self.model.to("cpu")
        return average_loss, accuracy.item()

    def update_best_model(self, test_acc, test_loss, round_id):
        # 模型选择规则：先比较 global_test_acc；acc 相同再比较 global_test_loss。
        is_better = (
            test_acc > self.best_test_acc
            or (test_acc == self.best_test_acc and test_loss < self.best_test_loss)
        )
        if not is_better:
            return False

        self.best_test_acc = test_acc
        self.best_test_loss = test_loss
        self.best_round = round_id
        self.best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        best_model_path = os.path.join(self.args.model_save_path, "best_server.pth")
        torch.save(
            {
                "model_state_dict": self.best_state_dict,
                "best_round": self.best_round,
                "best_test_acc": self.best_test_acc,
                "best_test_loss": self.best_test_loss,
            },
            best_model_path,
        )
        self.logger.info(
            f"--best_global_test_acc : {self.best_test_acc:.4f} "
            f"--best_global_test_loss : {self.best_test_loss:.4f} "
            f"--best_round : {self.best_round}\n"
        )
        return True

    def get_client_train_size(self, client_id):
        # sample_weighted 聚合会使用客户端训练样本数作为权重来源。
        return get_client_train_size(self.args, client_id, meta=self.partition_meta)

    def aggregation_by_method(self, client_states=None, uoc_evidence=None, reference_uoc_evidence=None, client_stats=None, round_index=None):
        # 聚合器接口：
        # - 非专家参数使用 non_expert_agg_method；
        # - 专家参数使用 expert_agg_method。
        client_sizes = []

        if client_states is None:
            loaded_client_states = []
            for id in self.clientsID_list:
                client_state_dict = torch.load(
                    os.path.join(self.args.model_save_path, f"{id}.pth"),
                    map_location="cpu",
                )
                loaded_client_states.append(client_state_dict)
                client_sizes.append(self.get_client_train_size(id))
            client_states = loaded_client_states
        else:
            for id in self.clientsID_list:
                client_sizes.append(self.get_client_train_size(id))

        use_uoc_foga = self.args.expert_agg_method in {
            "uoc_foga_expert_align",
            "uoc_foga_pism_expert_align",
        }
        if use_uoc_foga:
            # UOC-FOGA 需要在 global_model 上计算 g_query，让模型先到目标设备。
            self.model.to(self.device)

        try:
            aggregated_state = self.aggregator.aggregate(
                client_updates=client_states,
                client_weights=client_sizes,
                global_model=self.model,
                uoc_evidence=uoc_evidence,
                reference_uoc_evidence=reference_uoc_evidence,
                client_stats=client_stats,
                round_index=round_index,
            )
            self.model.load_state_dict(aggregated_state)
        finally:
            if use_uoc_foga and not bool(getattr(self.args, "keep_uoc_model_on_device_until_eval", True)):
                self.model.to("cpu")
        aggregation_metrics = getattr(self.aggregator, "last_aggregation_metrics", {})
        expert_aggregation_weights = aggregation_metrics.get("expert_aggregation_weights")
        if expert_aggregation_weights is not None:
            self.logger.info(
                "--expert_aggregation_weights : "
                f"{json.dumps(expert_aggregation_weights, ensure_ascii=False, sort_keys=True)}\n"
            )
        uoc_foga_stats = aggregation_metrics.get("uoc_foga_stats")
        pism_summary = aggregation_metrics.get("uoc_foga_pism_summary", None)
        if uoc_foga_stats is not None and bool(getattr(self.args, "uoc_foga_log_detail", False)):
            self.logger.info(f"--uoc_foga_stats : {uoc_foga_stats}\n")
        if uoc_foga_stats is not None:
            uoc_summary = self._summarize_uoc_foga_stats(uoc_foga_stats)
            if uoc_summary is not None:
                base_summary_keys = ()
                if pism_summary is None:
                    base_summary_keys = (
                        "uoc_foga_updated_experts",
                        "uoc_foga_fallback_experts",
                        "uoc_foga_fallback_reason_counts",
                        "uoc_foga_weight_entropy_mean",
                        "uoc_foga_score_mean_mean",
                    )
                query_summary_keys = (
                    "uoc_foga_score_metric",
                    "uoc_foga_query_select_mode",
                    "uoc_foga_query_fallback_to_random_count",
                    "uoc_foga_query_token_ratio_mean_mean",
                    "uoc_foga_query_entropy_mean_mean",
                    "uoc_foga_query_pool_after_ratio_filter_mean",
                )
                grad_dot_summary_keys = ()
                if uoc_summary.get("uoc_foga_score_metric") == "grad_dot":
                    grad_dot_summary_keys = (
                        "grad_dot_valid_scores",
                        "grad_dot_positive_frac",
                        "grad_dot_score_mean",
                        "grad_dot_score_std",
                        "grad_dot_score_min",
                        "grad_dot_score_max",
                        "grad_dot_client_set_size_mean",
                        "grad_dot_client_set_size_min",
                        "grad_dot_client_set_size_max",
                        "cos_delta_neg_gclient_mean",
                        "cos_delta_neg_gclient_positive_frac",
                    )
                delta_consensus_summary_keys = ()
                if uoc_summary.get("uoc_foga_score_metric") == "delta_consensus":
                    delta_consensus_summary_keys = (
                        "delta_consensus_valid_scores",
                        "delta_consensus_positive_frac",
                        "delta_consensus_score_mean",
                        "delta_consensus_score_std",
                        "delta_consensus_score_min",
                        "delta_consensus_score_max",
                        "delta_consensus_ref_clients_mean",
                    )
                for key in (
                    base_summary_keys
                    + query_summary_keys
                    + grad_dot_summary_keys
                    + delta_consensus_summary_keys
                ):
                    self.logger.info(f"--{key} : {uoc_summary.get(key)}\n")

        if pism_summary is not None:
            # PISM summary 只打印轻量标量/字典，不输出 per-expert 大对象。
            if pism_summary.get("uoc_foga_score_metric") == "grad_cosine":
                self.logger.info(
                    "[UOC-FOGA-PISM-FEATURE] "
                    f"score_metric={pism_summary.get('uoc_foga_score_metric')} "
                    f"input_dim={pism_summary.get('uoc_foga_pism_input_dim')} "
                    f"consensus_grad_cos_mean={pism_summary.get('uoc_foga_pism_consensus_grad_cos_mean')} "
                    f"consensus_grad_pos_frac_mean={pism_summary.get('uoc_foga_pism_consensus_grad_pos_frac_mean')} "
                    f"expert_loss_z_std_mean={pism_summary.get('uoc_foga_pism_expert_loss_z_std_mean')}\n"
                )
                self.logger.info(
                    "[UOC-FOGA-MIXED-QUERY] "
                    f"global_size_mean={pism_summary.get('uoc_foga_mixed_global_query_size_mean')} "
                    f"expert_size_mean={pism_summary.get('uoc_foga_mixed_expert_query_size_mean')} "
                    f"global_classes_mean={pism_summary.get('uoc_foga_mixed_global_query_num_classes_mean')} "
                    f"expert_classes_mean={pism_summary.get('uoc_foga_mixed_expert_query_num_classes_mean')} "
                    f"global_ratio_effective_mean={pism_summary.get('uoc_foga_mixed_global_ratio_effective_mean')}\n"
                )
            self.logger.info(
                "[UOC-FOGA-PISM-DIAG] "
                f"score_std_mean={pism_summary.get('uoc_foga_pism_score_std_mean')} "
                f"score_pos_frac_mean={pism_summary.get('uoc_foga_pism_score_pos_frac_mean')} "
                f"weight_score_corr_mean={pism_summary.get('uoc_foga_pism_weight_score_corr_mean')} "
                f"weight_score_corr_valid_frac={pism_summary.get('uoc_foga_pism_weight_score_corr_valid_frac')} "
                f"pism_top_score_rank_mean={pism_summary.get('uoc_foga_pism_pism_top_score_rank_mean')} "
                f"pism_top_score_value_mean={pism_summary.get('uoc_foga_pism_pism_top_score_value_mean')} "
                f"foga_top_pism_weight_mean={pism_summary.get('uoc_foga_pism_foga_top_pism_weight_mean')}\n"
            )
            self.logger.info(
                "[PISM-ALIGN-SUMMARY] "
                f"logit_score_corr_mean={pism_summary.get('uoc_foga_pism_logit_score_corr_mean')} "
                f"logit_score_corr_valid_frac={pism_summary.get('uoc_foga_pism_logit_score_corr_valid_frac')} "
                f"logit_std_mean={pism_summary.get('uoc_foga_pism_logit_std_mean')}\n"
            )
            for debug_record in pism_summary.get("pism_alignment_debug_records") or []:
                self.logger.info(
                    "[PISM-ALIGN-DEBUG] "
                    f"round={debug_record.get('round')} "
                    f"layer={debug_record.get('layer_id')} "
                    f"expert={debug_record.get('expert_id')} "
                    f"valid_clients={debug_record.get('valid_clients')} "
                    f"pism_top_score_rank={debug_record.get('pism_top_score_rank')} "
                    f"weight_score_corr={debug_record.get('weight_score_corr')} "
                    f"logit_score_corr={debug_record.get('logit_score_corr')}\n"
                )
                self.logger.info(
                    "--pism_alignment_debug_record : "
                    f"{json.dumps(debug_record, ensure_ascii=False, sort_keys=True)}\n"
                )
            self.logger.info(
                "[UOC-FOGA-QUERY-REF] "
                f"query_ref_cos_mean={pism_summary.get('uoc_foga_query_ref_cos_mean')} "
                f"query_ref_cos_std={pism_summary.get('uoc_foga_query_ref_cos_std')} "
                f"query_ref_cos_min={pism_summary.get('uoc_foga_query_ref_cos_min')} "
                f"query_ref_cos_valid_frac={pism_summary.get('uoc_foga_query_ref_cos_valid_frac')}\n"
            )
            self.logger.info(
                "[UOC-FOGA-SAMPLE-BIAS] "
                f"score_sample_corr_mean={pism_summary.get('uoc_foga_pism_score_sample_corr_mean')} "
                f"score_sample_corr_valid_frac={pism_summary.get('uoc_foga_pism_score_sample_corr_valid_frac')} "
                f"weight_sample_corr_mean={pism_summary.get('uoc_foga_pism_weight_sample_corr_mean')} "
                f"weight_sample_corr_valid_frac={pism_summary.get('uoc_foga_pism_weight_sample_corr_valid_frac')}\n"
            )
            self.logger.info(
                "[UOC-FOGA-REF-STEP] "
                f"foga_top_delta_mean={pism_summary.get('uoc_foga_ref_step_foga_top_loss_delta_mean')} "
                f"foga_top_improve_frac={pism_summary.get('uoc_foga_ref_step_foga_top_improve_frac')} "
                f"pism_top_delta_mean={pism_summary.get('uoc_foga_ref_step_pism_top_loss_delta_mean')} "
                f"pism_top_improve_frac={pism_summary.get('uoc_foga_ref_step_pism_top_improve_frac')} "
                f"valid_frac={pism_summary.get('uoc_foga_ref_step_valid_frac')}\n"
            )
            for key in (
                "uoc_foga_score_metric",
                "uoc_foga_pism_input_dim",
                "uoc_foga_pism_consensus_grad_cos_mean",
                "uoc_foga_pism_consensus_grad_pos_frac_mean",
                "uoc_foga_pism_expert_loss_z_std_mean",
                "uoc_foga_pism_logit_score_corr_mean",
                "uoc_foga_pism_logit_score_corr_valid_frac",
                "uoc_foga_pism_logit_std_mean",
                "uoc_foga_mixed_global_query_size_mean",
                "uoc_foga_mixed_expert_query_size_mean",
                "uoc_foga_mixed_global_query_num_classes_mean",
                "uoc_foga_mixed_expert_query_num_classes_mean",
                "uoc_foga_mixed_global_ratio_effective_mean",
                "uoc_foga_pism_meta_loss_mean",
                "uoc_foga_pism_updated_experts",
                "uoc_foga_pism_fallback_experts",
                "uoc_foga_pism_fallback_reason_counts",
                "uoc_foga_pism_weight_entropy_mean",
                "uoc_foga_pism_weight_max_mean",
                "uoc_foga_pism_min_weight_factor",
                "uoc_foga_pism_fairness_blend",
                "pism_post_weight_min_mean",
                "pism_post_weight_max_mean",
                "pism_post_weight_entropy_mean",
                "uoc_foga_pism_used_frac",
                "uoc_foga_pism_update_steps",
                "uoc_foga_pism_meta_steps",
                "uoc_foga_pism_meta_steps_successful",
                "uoc_foga_pism_meta_steps_requested",
                "uoc_foga_pism_tau_schedule",
                "uoc_foga_pism_tau",
                "uoc_foga_pism_tau_init",
                "uoc_foga_pism_tau_min",
                "uoc_foga_pism_tau_decay",
                "uoc_foga_query_ref_cos_mean",
                "uoc_foga_query_ref_cos_std",
                "uoc_foga_query_ref_cos_min",
                "uoc_foga_query_ref_cos_valid_frac",
                "uoc_foga_pism_score_sample_corr_mean",
                "uoc_foga_pism_score_sample_corr_valid_frac",
                "uoc_foga_pism_weight_sample_corr_mean",
                "uoc_foga_pism_weight_sample_corr_valid_frac",
                "uoc_foga_ref_step_foga_top_loss_delta_mean",
                "uoc_foga_ref_step_foga_top_improve_frac",
                "uoc_foga_ref_step_pism_top_loss_delta_mean",
                "uoc_foga_ref_step_pism_top_improve_frac",
                "uoc_foga_ref_step_valid_frac",
                "uoc_foga_client_grad_query_per_class",
                "uoc_foga_client_grad_min_samples_per_expert",
                "uoc_foga_client_grad_min_classes_per_expert",
                "uoc_foga_client_grad_min_expert_token_ratio",
                "uoc_foga_client_grad_max_samples_per_client_per_class",
                "uoc_foga_client_grad_fallback_to_random",
            ):
                self.logger.info(f"--{key} : {pism_summary.get(key)}\n")
        self.logger.info(
            f"--non_expert_agg_method : {self.args.non_expert_agg_method} "
            f"--expert_agg_method : {self.args.expert_agg_method}\n"
        )
        self.logger.info(f"--client_train_sizes : {client_sizes}\n")

    def aggregation(self, client_states=None, uoc_evidence=None, reference_uoc_evidence=None, client_stats=None, round_index=None):
        self.aggregation_by_method(
            client_states=client_states,
            uoc_evidence=uoc_evidence,
            reference_uoc_evidence=reference_uoc_evidence,
            client_stats=client_stats,
            round_index=round_index,
        )
