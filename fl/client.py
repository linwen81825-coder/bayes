import torch
import torch.optim as optim
from types import SimpleNamespace
from torch import nn

from data.loader import build_client_train_loader
from fl.bayes_utils import run_expert_sgld_fit
from model import build_model_from_args
from utils.utils import record_result

class Client:
    # Client 表示联邦学习里的一个客户端。
    # 每个客户端有自己的数据和模型，服务端每一轮会让多个客户端分别训练。
    def __init__(self, args: SimpleNamespace, client_id: int, logger, c_T: int, partition_meta=None):
        self.args = args
        self.client_id = client_id
        # 从 save/model/{client_id}.pth 加载这个客户端自己的模型。
        self.model = self.load_client_model()
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
        self.bayes_min_expert_tokens = getattr(self.args, "bayes_min_expert_tokens", 1)
        self.bayes_sgld_steps = max(int(getattr(self.args, "bayes_sgld_steps", 20)), 1)
        self.bayes_sgld_burnin = min(
            max(int(getattr(self.args, "bayes_sgld_burnin", 10)), 0),
            self.bayes_sgld_steps - 1,
        )
        self.bayes_sgld_lr = float(getattr(self.args, "bayes_sgld_lr", 0.00005))
        self.bayes_ai_max = float(getattr(self.args, "bayes_ai_max", 1e3))
        self.bayes_evidence_batches = max(int(getattr(self.args, "bayes_evidence_batches", 8)), 1)

    def load_client_model(self):
        # 客户端模型路径，例如 ./save/model/1.pth。
        self.model_path = self.args.model_save_path + f"/{self.client_id}.pth"
        model = build_model_from_args(self.args)
        state_dict = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def save_client_model(self):
        # 本地训练结束后，把客户端模型保存回原来的路径。
        cpu_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        torch.save(cpu_state_dict, self.model_path)

    def get_dataloader(self):
        # 客户端只拥有自己的训练数据；验证和测试都由服务端统一执行。
        self.train_loader = build_client_train_loader(
            args=self.args,
            client_id=self.client_id,
            meta=self.partition_meta,
        )


    def renew_model(self):
        # 每一轮本地训练前，客户端同步服务端 state_dict。
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

    def update_bayes_batch_cache(self, batch_cache_by_expert, inputs, labels, layer_stats):
        if not self.should_collect_bayes_evidence():
            return

        cpu_inputs = inputs.detach().cpu()
        cpu_labels = labels.detach().cpu()

        for layer_id, stats in layer_stats.items():
            sample_hits_by_expert = stats.get("sample_hits_by_expert")
            if sample_hits_by_expert is not None:
                sample_hits_by_expert = sample_hits_by_expert.detach().cpu()
                for expert_id, sample_hits in enumerate(sample_hits_by_expert):
                    sample_indices = torch.nonzero(sample_hits > 0, as_tuple=False).flatten()
                    if sample_indices.numel() == 0:
                        continue

                    expert_score = int(sample_hits[sample_indices].sum().item())
                    expert_inputs = cpu_inputs.index_select(0, sample_indices).clone()
                    expert_labels = cpu_labels.index_select(0, sample_indices).clone()
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
                expert_cache.append((expert_usage, cpu_inputs.clone(), cpu_labels.clone()))
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

    def fit_local_expert_evidence(self, layer_id, expert_id, usage, batch_cache):
        evidence_model = self.build_evidence_model()
        try:
            mean_state, precision_state = run_expert_sgld_fit(
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
            )
        finally:
            del evidence_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "usage": usage,
            "num_batches": len(batch_cache),
            "mean_state": mean_state,
            "precision_state": precision_state,
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
        evidence_by_layer = {}
        for layer_id, expert_id, usage in active_experts:
            batch_cache = self.get_expert_batch_cache(
                batch_cache_by_expert=batch_cache_by_expert,
                layer_id=layer_id,
                expert_id=expert_id,
            )
            if len(batch_cache) == 0:
                continue
            layer_evidence = evidence_by_layer.setdefault(layer_id, {})
            layer_evidence[expert_id] = self.fit_local_expert_evidence(
                layer_id=layer_id,
                expert_id=expert_id,
                usage=usage,
                batch_cache=batch_cache,
            )

        return evidence_by_layer

    def train(self):
        # 本地训练保持普通监督学习；不同模型通过 forward 返回的 aux loss / stats 接入路由约束和日志。
        self.renew_model()

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

        self.save_client_model()
        layer_stats_cpu = {
            layer_id: {
                stat_key: (value.detach().cpu() if torch.is_tensor(value) else value)
                for stat_key, value in stats.items()
            }
            for layer_id, stats in local_layer_usage_total.items()
        }
        bayes_evidence = self.extract_bayesian_evidence(layer_stats_cpu, bayes_batch_cache_by_expert)
        return {
            "expert_activations": local_usage_total.detach().cpu(),
            "expert_stats_by_layer": layer_stats_cpu,
            "expert_activations_by_layer": {
                layer_id: stats["expert_activations"]
                for layer_id, stats in layer_stats_cpu.items()
            },
            "bayes_evidence_by_layer": bayes_evidence,
        }
