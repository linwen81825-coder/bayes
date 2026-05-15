import math
import os
import time
from types import SimpleNamespace

import torch
from torch import nn

from data.loader import build_client_train_loader, build_global_eval_loader, get_client_train_size, load_partition_meta
from fl.aggregators import build_aggregator
from fl.client import Client
from model import build_model_from_args
from utils.utils import (
    estimate_eta,
    format_seconds,
    init_result_csv,
    init_server_result_csv,
    make_tqdm,
    record_server_result,
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
        self.clientsID_list = [i+1 for i in range(self.num_clients)]
        self.device = self.args.device
        self.resume = bool(getattr(self.args, "resume", False))
        self.start_round = 0
        self.resume_checkpoint_path = self.resolve_resume_checkpoint_path()
        self.save_client_models = bool(getattr(self.args, "save_client_models", False))
        self.empty_cache_after_round = bool(getattr(self.args, "empty_cache_after_round", False))
        # 服务端模型保存路径，例如 ./save/model/server.pth。
        self.model_path = self.args.model_save_path + f"/server.pth"
        self.bayes_state_path = os.path.join(self.args.model_save_path, "server_bayes_state.pth")
        self.logger = logger
        os.makedirs(self.args.model_save_path, exist_ok=True)
        self.partition_meta = load_partition_meta(self.args)
        self.global_test_loader = build_global_eval_loader(
            args=self.args,
            split="global_test",
            meta=self.partition_meta,
        )
        self.client_train_loaders = {
            client_id: build_client_train_loader(
                args=self.args,
                client_id=client_id,
                meta=self.partition_meta,
            )
            for client_id in self.clientsID_list
        }
        # 初始化全局模型和（如需要）Bayes 状态；resume 只在显式开关打开时触发。
        self.model = build_model_from_args(self.args)
        self.bayes_state = None
        if self.resume:
            self.load_resume_checkpoint()
        else:
            self.logger.info("--resume : False")
            self.logger.info("--start_round : 1")
            self.save_server_model()
            self.init_bayes_state()
        # 客户端初始模型直接来自同一个服务端模型，避免额外随机初始化。
        self.sync_clients_model()
        self.num_experts = self.args.num_experts
        self.criterion = nn.CrossEntropyLoss()
        self.last_client_expert_usages = []
        self.last_client_bayes_evidence = []
        self.last_client_model_states = []
        self.logger.info(f"--save_client_models : {self.save_client_models}")
        self.logger.info(f"--empty_cache_after_round : {self.empty_cache_after_round}")
        # 初始化 CSV 结果文件，后续客户端训练会不断追加记录。
        init_result_csv(self.args)
        init_server_result_csv(self.args)


    def init_global_model(self):
        # 保留旧接口，但不再隐式恢复旧 checkpoint。
        self.model = build_model_from_args(self.args)
        self.save_server_model()

    def save_server_model(self):
        # 保存当前服务端模型参数到 server.pth。
        cpu_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        torch.save(cpu_state_dict, self.args.model_save_path + f"/server.pth")

    def uses_bayesian_aggregation(self):
        return self.args.agg_method == "expert_bayes_meta"

    def resolve_resume_checkpoint_path(self):
        path = getattr(self.args, "resume_checkpoint_path", None)
        if path is None:
            return os.path.join(self.args.model_save_path, "resume_checkpoint.pth")
        path_str = str(path).strip()
        if path_str == "" or path_str.lower() == "null":
            return os.path.join(self.args.model_save_path, "resume_checkpoint.pth")
        return path_str

    def init_bayes_state(self):
        if not self.uses_bayesian_aggregation():
            self.bayes_state = None
            return

        self.bayes_state = self.build_initial_bayes_state()
        self.save_bayes_state()

    def load_resume_checkpoint(self):
        if os.path.exists(self.resume_checkpoint_path):
            checkpoint = torch.load(self.resume_checkpoint_path, map_location="cpu")
            completed_round = int(checkpoint.get("completed_round", 0))

            ckpt_agg_method = checkpoint.get("agg_method")
            if ckpt_agg_method is not None and ckpt_agg_method != self.args.agg_method:
                raise RuntimeError(
                    "Resume checkpoint agg_method mismatch: "
                    f"checkpoint agg_method={ckpt_agg_method!r}, "
                    f"current args.agg_method={self.args.agg_method!r}. "
                    "Please use the same aggregation method when resuming, or start a new run_name."
                )

            server_state = checkpoint.get("server_model_state_dict")
            if server_state is None:
                raise KeyError(
                    f"Missing server_model_state_dict in resume checkpoint: {self.resume_checkpoint_path}"
                )

            self.model.load_state_dict(server_state)
            self.bayes_state = checkpoint.get("bayes_state", None)

            if self.uses_bayesian_aggregation() and self.bayes_state is None:
                raise RuntimeError(
                    "resume=True with expert_bayes_meta requires bayes_state in resume_checkpoint.pth"
                )

            self.start_round = completed_round
            self.save_server_model()
            if self.uses_bayesian_aggregation():
                self.save_bayes_state()
            self.logger.info("--resume : True")
            self.logger.info("--resume_mode : checkpoint")
            self.logger.info(f"--resume_checkpoint_path : {self.resume_checkpoint_path}")
            self.logger.info(f"--resume_completed_round : {completed_round}")
            self.logger.info(f"--resume_start_round : {self.start_round + 1}")
            self.logger.info(f"--target_server_epochs : {self.server_epochs}")
            return

        if bool(getattr(self.args, "resume_allow_legacy_checkpoint", True)):
            self.load_legacy_resume_checkpoint()
            return

        raise FileNotFoundError(
            f"resume=True but resume checkpoint not found: {self.resume_checkpoint_path}"
        )

    def infer_completed_round_from_server_csv(self):
        import csv

        from utils.utils import get_server_csv_path

        csv_path = get_server_csv_path(self.args)
        if not os.path.exists(csv_path):
            return 0

        completed_round = 0
        with open(csv_path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                phase = str(row.get("phase", "")).strip().lower()
                if phase != "test":
                    continue
                try:
                    round_id = int(float(row.get("round", 0)))
                except (TypeError, ValueError):
                    continue
                completed_round = max(completed_round, round_id)
        return completed_round

    def infer_best_acc_from_server_csv(self):
        import csv

        from utils.utils import get_server_csv_path

        csv_path = get_server_csv_path(self.args)
        if not os.path.exists(csv_path):
            return float("-inf")

        best_acc = float("-inf")
        with open(csv_path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                phase = str(row.get("phase", "")).strip().lower()
                if phase != "test":
                    continue
                try:
                    acc = float(row.get("test_acc"))
                except (TypeError, ValueError):
                    continue
                best_acc = max(best_acc, acc)
        return best_acc

    def load_legacy_resume_checkpoint(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"resume=True legacy fallback requires server model: {self.model_path}"
            )

        server_state = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(server_state)

        self.bayes_state = None
        if self.uses_bayesian_aggregation():
            if not os.path.exists(self.bayes_state_path):
                raise FileNotFoundError(
                    "resume=True with expert_bayes_meta requires server_bayes_state.pth "
                    f"for legacy fallback, but not found: {self.bayes_state_path}"
                )
            self.bayes_state = torch.load(self.bayes_state_path, map_location="cpu")

        completed_round = self.infer_completed_round_from_server_csv()
        if self.uses_bayesian_aggregation() and self.bayes_state is not None:
            bayes_round = int(self.bayes_state.get("round", 0))
            if completed_round > 0 and bayes_round > 0 and completed_round != bayes_round:
                raise RuntimeError(
                    "Legacy Bayes resume state is inconsistent: "
                    f"server_csv_completed_round={completed_round}, "
                    f"bayes_state_round={bayes_round}. "
                    "This may mean the previous run was interrupted during a round. "
                    "Please inspect server.pth, server_bayes_state.pth and server CSV manually, "
                    "or resume from resume_checkpoint.pth if available."
                )
            if completed_round <= 0 and bayes_round > 0:
                completed_round = bayes_round

        self.start_round = int(completed_round)
        self.save_server_model()
        if self.uses_bayesian_aggregation():
            self.save_bayes_state()

        self.logger.info("--resume : True")
        self.logger.info("--resume_mode : legacy_fallback")
        self.logger.info(f"--resume_server_model : {self.model_path}")
        if self.uses_bayesian_aggregation():
            self.logger.info(f"--resume_bayes_state : {self.bayes_state_path}")
        self.logger.info(f"--resume_completed_round : {self.start_round}")
        self.logger.info(f"--resume_start_round : {self.start_round + 1}")
        self.logger.info(f"--target_server_epochs : {self.server_epochs}")

    def build_initial_bayes_state(self):
        gamma0_init = max(float(getattr(self.args, "bayes_gamma0_init", 1.0)), 1e-8)
        n0_init = max(float(getattr(self.args, "bayes_n0_init", 1.0)), 1e-8)
        log_gamma0 = math.log(gamma0_init)
        log_n0 = math.log(n0_init)
        bayes_state = {
            "round": 0,
            "gamma0_init": gamma0_init,
            "n0_init": n0_init,
            "experts": {},
        }

        for key, value in self.model.state_dict().items():
            expert_ref = self.parse_expert_ref(key)
            if expert_ref is None:
                continue

            layer_id, expert_id = expert_ref
            layer_state = bayes_state["experts"].setdefault(layer_id, {})
            expert_state = layer_state.setdefault(
                expert_id,
                {
                    "log_precision_state": {},
                    "log_n0": torch.tensor(log_n0, dtype=torch.float32),
                },
            )
            if torch.is_floating_point(value):
                expert_state["log_precision_state"][key] = torch.full_like(
                    value.detach().cpu(),
                    fill_value=log_gamma0,
                )
            else:
                expert_state["log_precision_state"][key] = value.detach().cpu().clone()

        return bayes_state

    def save_bayes_state(self):
        if self.bayes_state is None:
            return
        torch.save(self.bayes_state, self.bayes_state_path)

    def save_resume_checkpoint(self, completed_round: int):
        os.makedirs(self.args.model_save_path, exist_ok=True)
        resume_dir = os.path.dirname(self.resume_checkpoint_path)
        if resume_dir:
            os.makedirs(resume_dir, exist_ok=True)
        checkpoint = {
            "completed_round": int(completed_round),
            "server_model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            },
            "bayes_state": self.bayes_state,
            "args_snapshot": vars(self.args).copy(),
            "agg_method": self.args.agg_method,
        }
        torch.save(checkpoint, self.resume_checkpoint_path)
        self.logger.info(
            f"--save_resume_checkpoint : {self.resume_checkpoint_path} "
            f"--completed_round : {int(completed_round)}"
        )

    def parse_expert_ref(self, key):
        parts = key.split(".")
        if "blocks" not in parts or "experts" not in parts:
            return None

        blocks_idx = parts.index("blocks")
        experts_idx = parts.index("experts")
        if blocks_idx + 1 >= len(parts) or experts_idx + 1 >= len(parts):
            return None
        if not parts[blocks_idx + 1].isdigit() or not parts[experts_idx + 1].isdigit():
            return None

        return parts[blocks_idx + 1], parts[experts_idx + 1]

    def count_bayes_evidence_entries(self, evidence_by_layer):
        total = 0
        for expert_map in evidence_by_layer.values():
            total += len(expert_map)
        return total

    def unpack_aggregation_output(self, aggregation_output):
        if hasattr(aggregation_output, "model_state"):
            model_state = aggregation_output.model_state
            bayes_state = getattr(aggregation_output, "bayes_state", None)
            metrics = getattr(aggregation_output, "metrics", {}) or {}
            return model_state, bayes_state, metrics

        if isinstance(aggregation_output, dict) and "model_state" in aggregation_output:
            model_state = aggregation_output["model_state"]
            bayes_state = aggregation_output.get("bayes_state")
            metrics = aggregation_output.get("metrics", {}) or {}
            return model_state, bayes_state, metrics

        return aggregation_output, None, {}

    def sync_clients_model(self):
        if not self.save_client_models:
            self.logger.info("--sync_clients_model : skipped client checkpoints because save_client_models=False")
            return

        server_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        for id in self.clientsID_list:
            # 初始化时所有客户端文件都保存同一个服务端 state_dict。
            model_path = self.args.model_save_path + f"/{id}.pth"
            torch.save(server_state_dict, model_path)




    def train(self):
        # 外层循环是一轮轮服务端通信，也就是联邦学习中的 global round。
        training_start = time.perf_counter()
        best_acc = self.infer_best_acc_from_server_csv() if self.resume else float("-inf")
        last_acc = None
        if self.start_round >= self.server_epochs:
            self.logger.info(
                f"[Resume] completed_round={self.start_round} >= "
                f"server_epochs={self.server_epochs}; nothing to train."
            )
            return

        remaining_rounds = self.server_epochs - self.start_round
        progress_steps_per_round = len(self.clientsID_list) + 1
        progress_total_steps = remaining_rounds * progress_steps_per_round
        progress_iter = make_tqdm(
            range(progress_total_steps),
            self.args,
            desc=f"Training[{self.args.agg_method}]",
            total=progress_total_steps,
            leave=True,
        )
        try:
            for c_T in range(self.start_round, self.server_epochs):
                round_start = time.perf_counter()
                self.logger.info(f"============================== T:{c_T+1} start !!! ===============================\n")
                round_expert_usage_summary = torch.zeros(self.args.num_experts)
                round_layer_stats = {}
                round_client_expert_usages = []
                round_client_bayes_evidences = []
                round_client_model_states = []
                round_client_total = 0.0
                round_client_train_total = 0.0
                round_client_evidence_total = 0.0
                round_client_overhead_total = 0.0
                round_aggregation_time = 0.0
                round_eval_time = 0.0
                server_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                for client_index, id in enumerate(self.clientsID_list, start=1):
                    # 每个客户端执行本地训练，并返回本轮信息。
                    client_start = time.perf_counter()
                    client_stats = Client(
                        args=self.args,
                        client_id=id,
                        logger=self.logger,
                        c_T=c_T,
                        partition_meta=self.partition_meta,
                        train_loader=self.client_train_loaders.get(id),
                        server_state_dict=server_state_dict,
                    ).train()
                    client_time = time.perf_counter() - client_start
                    local_train_time = float(client_stats.get("local_train_time", 0.0) or 0.0)
                    bayes_evidence_time = float(client_stats.get("bayes_evidence_time", 0.0) or 0.0)
                    client_overhead_time = max(client_time - local_train_time - bayes_evidence_time, 0.0)
                    round_client_total += client_time
                    round_client_train_total += local_train_time
                    round_client_evidence_total += bayes_evidence_time
                    round_client_overhead_total += client_overhead_time
                    client_expert_usage = client_stats["expert_activations"].float().cpu()
                    round_client_expert_usages.append(
                        {
                            "expert_activations": client_stats["expert_activations"],
                            "expert_stats_by_layer": client_stats.get("expert_stats_by_layer", {}),
                            "expert_activations_by_layer": client_stats.get("expert_activations_by_layer", {}),
                        }
                    )
                    round_client_bayes_evidences.append(client_stats.get("bayes_evidence_by_layer", {}))
                    round_client_model_states.append(client_stats["local_state_dict"])
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
                    if hasattr(progress_iter, "set_postfix_str"):
                        progress_iter.set_postfix_str(
                            f"r={c_T + 1}/{self.server_epochs}, "
                            f"c={client_index}/{len(self.clientsID_list)}, "
                            f"tr={local_train_time:.1f}s",
                            refresh=False,
                        )
                    if hasattr(progress_iter, "update"):
                        progress_iter.update(1)

                usage_list = [int(v) for v in round_expert_usage_summary.tolist()]
                self.logger.info(f"--round_expert_usage_summary : {usage_list}\n")
                self.last_client_expert_usages = round_client_expert_usages
                self.last_client_bayes_evidence = round_client_bayes_evidences
                self.last_client_model_states = round_client_model_states
                client_usage_list = [
                    [int(v) for v in stats["expert_activations"].tolist()]
                    for stats in round_client_expert_usages
                ]
                client_bayes_counts = [
                    self.count_bayes_evidence_entries(evidence)
                    for evidence in round_client_bayes_evidences
                ]
                layer_stats_log = {
                    layer_id: {
                        "expert_activations": [int(v) for v in stats["expert_activations"].tolist()],
                        "overflow_counts": [int(v) for v in stats["overflow_counts"].tolist()],
                        "capacity": int(stats["capacity"]),
                    }
                    for layer_id, stats in round_layer_stats.items()
                }
                self.logger.info(f"--client_expert_usage_summary : {client_usage_list}\n")
                self.logger.info(f"--round_expert_stats_by_layer : {layer_stats_log}\n")
                self.logger.info(f"--client_bayes_evidence_counts : {client_bayes_counts}\n")
                # 所有客户端本地训练完成后，服务端通过聚合器更新全局模型。
                aggregation_start = time.perf_counter()
                self.aggregation()
                round_aggregation_time = time.perf_counter() - aggregation_start

                # 不再单独划分验证集；服务端每轮直接在官方 test set 上测试当前全局模型。
                eval_start = time.perf_counter()
                test_loss, test_acc = self.evaluate_global_model(self.global_test_loader)
                round_eval_time = time.perf_counter() - eval_start
                if test_acc > best_acc:
                    best_acc = test_acc
                last_acc = test_acc
                self.logger.info(f"--server_global_test_loss : {test_loss:.4f} --server_global_test_acc : {test_acc:.4f}\n")
                record_server_result(
                    {
                        "phase": "test",
                        "round": c_T + 1,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                    },
                    self.args,
                )

                # 每轮结束保存当前服务端模型，供下一轮客户端同步。
                self.save_server_model()
                self.save_resume_checkpoint(completed_round=c_T + 1)
                if (
                    self.empty_cache_after_round
                    and str(self.device).startswith("cuda")
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

                round_elapsed = time.perf_counter() - round_start
                elapsed = time.perf_counter() - training_start
                completed_this_run = c_T + 1 - self.start_round
                eta, avg_round_time = estimate_eta(elapsed, completed_this_run, remaining_rounds)
                if hasattr(progress_iter, "update"):
                    progress_iter.update(1)
                progress_summary = (
                    f"[progress] round={c_T + 1}/{self.server_epochs} "
                    f"elapsed={format_seconds(elapsed)} "
                    f"eta={format_seconds(eta)} "
                    f"last_round={format_seconds(round_elapsed)} "
                    f"avg_round={format_seconds(avg_round_time)} "
                    f"train={format_seconds(round_client_train_total)} "
                    f"evidence={format_seconds(round_client_evidence_total)} "
                    f"client_total={format_seconds(round_client_total)} "
                    f"client_overhead={format_seconds(round_client_overhead_total)} "
                    f"aggregation={format_seconds(round_aggregation_time)} "
                    f"eval={format_seconds(round_eval_time)} "
                    f"acc={test_acc:.4f} "
                    f"best={best_acc:.4f} "
                    f"agg={self.args.agg_method}"
                )
                self.logger.info(progress_summary)
        finally:
            if hasattr(progress_iter, "close"):
                progress_iter.close()

    def evaluate_global_model(self, data_loader):
        self.model.to(self.device)
        self.model.eval()
        running_loss = torch.zeros((), device=self.device)
        running_corrects = torch.zeros((), device=self.device)
        non_blocking = bool(getattr(self.args, "pin_memory", False)) and str(self.device).startswith("cuda")

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device, non_blocking=non_blocking)
                labels = labels.to(self.device, non_blocking=non_blocking)
                result = self.model(inputs)
                outputs = result["logits"]
                loss = self.criterion(outputs, labels)

                running_loss += loss.detach() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        eval_loss = (running_loss / len(data_loader.dataset)).item()
        eval_acc = (running_corrects.double() / len(data_loader.dataset)).item()
        self.model.to("cpu")
        return eval_loss, eval_acc

    def get_client_train_size(self,client_id):
        # FedAvg 使用客户端训练样本数作为聚合权重。
        return get_client_train_size(self.args, client_id, meta=self.partition_meta)

    def aggregation_by_method(self):
        # 聚合器接口：
        # - fedavg：对完整 state_dict 按客户端训练样本数加权平均；
        # - expert_fedavg：普通层按客户端样本数聚合，专家层按每个 expert 实际处理样本数聚合。
        # - expert_bayes_meta：在 expert 粒度额外接收客户端上传的局部贝叶斯证据和服务端先验状态。
        client_states = []
        client_sizes = []
        cached_client_states = getattr(self, "last_client_model_states", None)
        use_returned_states = (
            isinstance(cached_client_states, list)
            and len(cached_client_states) == len(self.clientsID_list)
        )
        for index, id in enumerate(self.clientsID_list):
            if use_returned_states:
                client_state_dict = cached_client_states[index]
            else:
                client_state_dict = torch.load(
                    self.args.model_save_path + f"/{id}.pth",
                    map_location="cpu",
                )
            client_states.append(client_state_dict)
            client_sizes.append(self.get_client_train_size(id))

        total_size = sum(client_sizes)
        if total_size <= 0:
            raise ValueError("FedAvg requires at least one training sample across clients")

        aggregation_output = self.aggregator.aggregate(
            client_updates=client_states,
            client_weights=client_sizes,
            global_model=self.model,
            expert_weights=getattr(self, "last_client_expert_usages", None),
            expert_evidence=getattr(self, "last_client_bayes_evidence", None),
            bayes_state=self.bayes_state,
        )
        aggregated_state, updated_bayes_state, aggregation_metrics = self.unpack_aggregation_output(
            aggregation_output
        )
        self.model.load_state_dict(aggregated_state)
        if updated_bayes_state is not None:
            self.bayes_state = updated_bayes_state
        self.save_bayes_state()
        self.logger.info(f"--aggregation_method : {self.args.agg_method}\n")
        self.logger.info(f"--client_train_sizes : {client_sizes}\n")
        if self.uses_bayesian_aggregation():
            total_evidence = sum(
                self.count_bayes_evidence_entries(evidence)
                for evidence in getattr(self, "last_client_bayes_evidence", [])
            )
            self.logger.info(f"--round_bayes_evidence_total : {total_evidence}\n")
        if aggregation_metrics:
            expert_meta_stats = aggregation_metrics.get("expert_meta_stats")
            summary_metrics = {
                key: value
                for key, value in aggregation_metrics.items()
                if key != "expert_meta_stats"
            }
            self.logger.info(f"--aggregation_metrics : {summary_metrics}\n")
            if expert_meta_stats:
                self.logger.info(f"--expert_meta_stats : {expert_meta_stats}\n")

    def aggregation(self):
        self.aggregation_by_method()

