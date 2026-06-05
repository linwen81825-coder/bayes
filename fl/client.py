import torch
import torch.optim as optim
from types import SimpleNamespace
from torch import nn

from data.loader import build_client_train_loader
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
        initial_state_dict=None,
        save_model_to_disk=True,
    ):
        self.args = args
        self.client_id = client_id
        self.model_path = self.args.model_save_path + f"/{self.client_id}.pth"
        self.save_model_to_disk = save_model_to_disk
        # 内存模式下直接加载服务端传入的 state_dict，旧模式下仍从 pth 读取。
        self.model = self.load_client_model(initial_state_dict=initial_state_dict)
        self.device = self.args.device
        self.model.to(self.device)
        # c_T 表示当前是第几轮服务端通信轮次，主要用于记录日志。
        self.c_T =  c_T
        self.client_epochs = self.args.client_epochs
        # 分类任务常用交叉熵损失。
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        self.batch_size = self.args.batch_size
        self.partition_meta = partition_meta
        self.train_loader = None
        # 加载当前客户端的训练索引，并动态封装成 DataLoader。
        self.get_dataloader()

        self.logger = logger
        self.router_aux_loss_coef = self.args.router_aux_loss_coef
        self.router_z_loss_coef = self.args.router_z_loss_coef

    def load_client_model(self, initial_state_dict=None):
        model = build_model_from_args(self.args)
        if initial_state_dict is not None:
            model.load_state_dict(initial_state_dict)
            return model

        state_dict = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def save_client_model(self):
        # 本地训练结束后，把客户端模型保存回原来的路径。
        torch.save(self.model.state_dict(), self.model_path)

    def get_dataloader(self):
        # 客户端只拥有自己的训练数据；验证和测试都由服务端统一执行。
        self.train_loader = build_client_train_loader(
            args=self.args,
            client_id=self.client_id,
            meta=self.partition_meta,
        )

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

    def _should_collect_uoc_evidence_before_train(self):
        expert_method = getattr(self.args, "expert_agg_method", "")
        uoc_foga_enabled = bool(getattr(self.args, "uoc_foga_enabled", False))
        uoc_foga_collect_before_train = bool(
            getattr(self.args, "uoc_foga_collect_before_train", True)
        )
        if not uoc_foga_collect_before_train:
            return False

        # UOC evidence 由 expert_agg_method 或手动开关触发，不再依赖 agg_method。
        return (
            expert_method in ["uoc_foga_expert_align", "uoc_foga_pism_expert_align"]
            or uoc_foga_enabled
        )

    def _collect_uoc_evidence_before_train(self):
        # 只在本地训练前，从 round-start global model 采集少量 UOC evidence。
        if not self._should_collect_uoc_evidence_before_train():
            return {}
        if not hasattr(self.model, "collect_uoc_evidence"):
            return {}

        samples_per_client = int(getattr(self.args, "uoc_foga_samples_per_client", 64))
        uoc_foga_use_top1 = bool(getattr(self.args, "uoc_foga_use_top1", True))
        if samples_per_client <= 0:
            return {}

        was_training = self.model.training
        self.model.eval()
        try:
            evidence_chunks_by_layer = {}
            collected_samples = 0
            evidence_loader = iter(self.train_loader)
            inference_context = (
                torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
            )

            for images, labels in evidence_loader:
                remaining = samples_per_client - collected_samples
                if remaining <= 0:
                    break

                images = images[:remaining]
                labels = labels[:remaining]
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if labels.size(0) == 0:
                    continue

                with inference_context():
                    batch_evidence = self.model.collect_uoc_evidence(
                        images,
                        max_samples=remaining,
                        use_top1=uoc_foga_use_top1,
                    )

                batch_sample_count = labels.size(0)
                for layer_id, layer_evidence in batch_evidence.items():
                    layer_key = str(layer_id)
                    hidden = layer_evidence["hidden"].detach()
                    top1_expert_ids = layer_evidence["top1_expert_ids"].detach()
                    top1_gates = layer_evidence["top1_gates"].detach()
                    residual = layer_evidence.get("residual")
                    if residual is not None:
                        # residual 用于 server 端精确恢复当前 MoE block 输出：
                        # x_after_block = residual + forced_expert(hidden)。
                        residual = residual[: hidden.size(0)].detach()
                    layer_labels = labels[: hidden.size(0)].detach()

                    if layer_key not in evidence_chunks_by_layer:
                        evidence_chunks_by_layer[layer_key] = {
                            "hidden": [],
                            "labels": [],
                            "top1_expert_ids": [],
                            "top1_gates": [],
                        }
                    if residual is not None and "residual" not in evidence_chunks_by_layer[layer_key]:
                        evidence_chunks_by_layer[layer_key]["residual"] = []

                    evidence_chunks_by_layer[layer_key]["hidden"].append(hidden.cpu())
                    evidence_chunks_by_layer[layer_key]["labels"].append(layer_labels.cpu())
                    evidence_chunks_by_layer[layer_key]["top1_expert_ids"].append(
                        top1_expert_ids.cpu()
                    )
                    evidence_chunks_by_layer[layer_key]["top1_gates"].append(
                        top1_gates.cpu()
                    )
                    if residual is not None:
                        evidence_chunks_by_layer[layer_key]["residual"].append(residual.cpu())

                collected_samples += batch_sample_count

            uoc_evidence_by_layer = {
                layer_id: {
                    stat_key: torch.cat(chunks, dim=0).detach().cpu()
                    for stat_key, chunks in layer_chunks.items()
                }
                for layer_id, layer_chunks in evidence_chunks_by_layer.items()
            }

            if uoc_evidence_by_layer and self.logger is not None:
                counts_by_layer = {
                    layer_id: int(layer_evidence["hidden"].size(0))
                    for layer_id, layer_evidence in uoc_evidence_by_layer.items()
                }
                self.logger.info(
                    f"[UOCEvidence] client={self.client_id} counts_by_layer={counts_by_layer}"
                )

            return uoc_evidence_by_layer
        finally:
            # evidence 采集临时使用 eval，结束后恢复原来的训练状态。
            self.model.train(was_training)

    def train(self):
        # 本地训练保持普通监督学习；不同模型通过 forward 返回的 aux loss / stats 接入路由约束和日志。
        non_blocking = (
            str(self.device).startswith("cuda")
            and bool(getattr(self.args, "pin_memory", False))
        )
        uoc_evidence_by_layer = self._collect_uoc_evidence_before_train()
        last_avg_router_probs = torch.zeros(self.args.num_experts, device=self.device)
        local_usage_total = torch.zeros(self.args.num_experts, device=self.device)
        local_layer_usage_total = {}
        round_loss_total = 0.0
        round_corrects = torch.zeros((), device=self.device)
        round_total_samples = 0

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
                inputs = inputs.to(self.device, non_blocking=non_blocking)
                labels = labels.to(self.device, non_blocking=non_blocking)
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
                self.add_layer_stats(layer_usage_total, self.get_layer_expert_stats(result))
                router_prob_sum += self.get_avg_router_probs(result) * batch_size

            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = running_corrects.double() / len(self.train_loader.dataset)
            round_loss_total += running_loss
            round_corrects += running_corrects.detach()
            round_total_samples += len(self.train_loader.dataset)
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

        if self.save_model_to_disk:
            self.save_client_model()
        round_train_loss = round_loss_total / max(round_total_samples, 1)
        round_train_acc = (round_corrects.double() / max(round_total_samples, 1)).item()
        layer_stats_cpu = {
            layer_id: {
                stat_key: (value.detach().cpu() if torch.is_tensor(value) else value)
                for stat_key, value in stats.items()
            }
            for layer_id, stats in local_layer_usage_total.items()
        }
        result = {
            "expert_activations": local_usage_total.detach().cpu(),
            "expert_stats_by_layer": layer_stats_cpu,
            "expert_activations_by_layer": {
                layer_id: stats["expert_activations"]
                for layer_id, stats in layer_stats_cpu.items()
            },
            "uoc_evidence_by_layer": uoc_evidence_by_layer,
            "train_loss": round_train_loss,
            "client_loss": round_train_loss,
            "train_acc": round_train_acc,
        }

        if not self.save_model_to_disk:
            result["model_state_dict"] = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }

        return result
