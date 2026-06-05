import os
from types import SimpleNamespace

import torch
from torch import nn
from tqdm import tqdm

from data.loader import build_global_eval_loader, get_client_train_size, load_partition_meta
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
        self.num_experts = self.args.num_experts
        self.criterion = nn.CrossEntropyLoss()

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
        )

        try:
            # 外层循环是一轮轮服务端通信，也就是联邦学习中的 global round。
            for c_T in range(self.start_round, self.server_epochs):
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
                for id in self.clientsID_list:
                    # 每个客户端执行本地训练，并返回本轮信息。
                    client_stats = Client(
                        args=self.args,
                        client_id=id,
                        logger=self.logger,
                        c_T=c_T,
                        partition_meta=self.partition_meta,
                        initial_state_dict=server_state_dict if use_in_memory_updates else None,
                        save_model_to_disk=not use_in_memory_updates,
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
                self.aggregation(
                    client_states=client_states if use_in_memory_updates else None,
                    uoc_evidence=round_client_uoc_evidences,
                    client_stats=round_client_stats,
                )

                test_loss, test_acc = self.evaluate_global_model(self.global_test_loader)
                self.logger.info(f"--server_global_test_loss : {test_loss:.4f} --server_global_test_acc : {test_acc:.4f}\n")
                is_best = self.update_best_model(test_acc=test_acc, test_loss=test_loss, round_id=c_T + 1)
                record_server_result(
                    {
                        "phase": "test",
                        "round": c_T + 1,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "best_test_acc": self.best_test_acc,
                        "best_test_loss": self.best_test_loss,
                        "is_best": int(is_best),
                    },
                    self.args,
                )

                # 每轮结束保存当前服务端模型，供下一轮客户端同步和断点续训。
                self.save_server_model()
                round_completed = c_T + 1
                self.save_training_checkpoint(round_completed)
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

        with torch.no_grad():
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

    def aggregation_by_method(self, client_states=None, uoc_evidence=None, client_stats=None):
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
                client_stats=client_stats,
            )
            self.model.load_state_dict(aggregated_state)
        finally:
            if use_uoc_foga:
                self.model.to("cpu")
        aggregation_metrics = getattr(self.aggregator, "last_aggregation_metrics", {})
        uoc_foga_stats = aggregation_metrics.get("uoc_foga_stats")
        if uoc_foga_stats is not None:
            self.logger.info(f"--uoc_foga_stats : {uoc_foga_stats}\n")

        pism_summary = aggregation_metrics.get("uoc_foga_pism_summary", None)
        if pism_summary is not None:
            # PISM summary 只打印轻量标量/字典，不输出 per-expert 大对象。
            for key in (
                "uoc_foga_pism_meta_loss_mean",
                "uoc_foga_pism_updated_experts",
                "uoc_foga_pism_fallback_experts",
                "uoc_foga_pism_fallback_reason_counts",
                "uoc_foga_pism_weight_entropy_mean",
                "uoc_foga_pism_weight_max_mean",
                "uoc_foga_pism_used_frac",
                "uoc_foga_pism_update_steps",
            ):
                self.logger.info(f"--{key} : {pism_summary.get(key)}\n")
        self.logger.info(
            f"--non_expert_agg_method : {self.args.non_expert_agg_method} "
            f"--expert_agg_method : {self.args.expert_agg_method}\n"
        )
        self.logger.info(f"--client_train_sizes : {client_sizes}\n")

    def aggregation(self, client_states=None, uoc_evidence=None, client_stats=None):
        self.aggregation_by_method(
            client_states=client_states,
            uoc_evidence=uoc_evidence,
            client_stats=client_stats,
        )
