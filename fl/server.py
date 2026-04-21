import torch
import os
from types import SimpleNamespace
from torch import nn

from data.loader import build_global_eval_loader, get_client_train_size, load_partition_meta
from fl.aggregators import build_aggregator
from fl.client import Client
from model import build_model_from_args
from utils.utils import init_result_csv, init_server_result_csv, record_server_result

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
        # 服务端模型保存路径，例如 ./save/model/server.pth。
        self.model_path = self.args.model_save_path + f"/server.pth"
        self.logger = logger
        os.makedirs(self.args.model_save_path, exist_ok=True)
        self.partition_meta = load_partition_meta(self.args)
        self.global_val_loader = build_global_eval_loader(
            args=self.args,
            split="global_val",
            meta=self.partition_meta,
        )
        self.global_test_loader = build_global_eval_loader(
            args=self.args,
            split="global_test",
            meta=self.partition_meta,
        )
        # 初始化全局模型，并保存到 server.pth。
        self.init_global_model()
        # 客户端初始模型直接来自同一个服务端模型，避免额外随机初始化。
        self.sync_clients_model()
        self.num_experts = self.args.num_experts
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = -1.0
        self.best_val_loss = float("inf")
        self.best_round = 0
        self.best_state_dict = None
        # 初始化 CSV 结果文件，后续客户端训练会不断追加记录。
        init_result_csv(self.args)
        init_server_result_csv(self.args)


    def init_global_model(self):
        # 根据 model_type 初始化全局模型。
        self.model = build_model_from_args(self.args)
        # 初始化完成后立即保存，客户端 renew_model 时会读取这个文件。
        self.save_server_model()

    def save_server_model(self):
        # 保存当前服务端模型参数到 server.pth。
        torch.save(self.model.state_dict(), self.args.model_save_path + f"/server.pth")


    def sync_clients_model(self):
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
        for c_T in range(self.server_epochs):
            self.logger.info(f"============================== T:{c_T+1} start !!! ===============================\n")
            round_expert_usage_summary = torch.zeros(self.args.num_experts)
            round_layer_stats = {}
            round_client_expert_usages = []
            for id in self.clientsID_list:
                # 每个客户端执行本地训练，并返回本轮信息。
                client_stats = Client(
                    args=self.args,
                    client_id=id,
                    logger=self.logger,
                    c_T=c_T,
                    partition_meta=self.partition_meta,
                ).train()
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
            self.last_client_expert_usages = round_client_expert_usages
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
            self.logger.info(f"--client_expert_usage_summary : {client_usage_list}\n")
            self.logger.info(f"--round_expert_stats_by_layer : {layer_stats_log}\n")
            # 所有客户端本地训练完成后，服务端通过聚合器更新全局模型。
            self.aggregation()

            # global_val 用于模型选择；global_test 只在训练结束后评估一次。
            val_loss, val_acc = self.evaluate_global_model(self.global_val_loader)
            self.logger.info(f"--server_global_val_loss : {val_loss:.4f} --server_global_val_acc : {val_acc:.4f}\n")
            is_best = self.update_best_model(val_acc=val_acc, val_loss=val_loss, round_id=c_T + 1)
            record_server_result(
                {
                    "phase": "val",
                    "round": c_T + 1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_acc": self.best_val_acc,
                    "best_val_loss": self.best_val_loss,
                    "is_best": int(is_best),
                    "selected_for_test": 0,
                },
                self.args,
            )

            # 每轮结束保存当前服务端模型，供下一轮客户端同步。
            self.save_server_model()
            torch.cuda.empty_cache()

        self.evaluate_best_on_global_test()

    def evaluate_global_model(self, data_loader):
        self.model.to(self.device)
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                result = self.model(inputs)
                outputs = result["logits"]
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        eval_loss = running_loss / len(data_loader.dataset)
        eval_acc = running_corrects.double() / len(data_loader.dataset)
        self.model.to("cpu")
        return eval_loss, eval_acc.item()

    def update_best_model(self, val_acc, val_loss, round_id):
        # 模型选择规则：先比较 global_val_acc；acc 相同再比较 global_val_loss。
        is_better = (
            val_acc > self.best_val_acc
            or (val_acc == self.best_val_acc and val_loss < self.best_val_loss)
        )
        if not is_better:
            return False

        self.best_val_acc = val_acc
        self.best_val_loss = val_loss
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
                "best_val_acc": self.best_val_acc,
                "best_val_loss": self.best_val_loss,
            },
            best_model_path,
        )
        self.logger.info(
            f"--best_global_val_acc : {self.best_val_acc:.4f} "
            f"--best_global_val_loss : {self.best_val_loss:.4f} "
            f"--best_round : {self.best_round}\n"
        )
        return True

    def evaluate_best_on_global_test(self):
        last_server_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

        if self.best_state_dict is None:
            self.best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }
            self.best_round = self.server_epochs

        self.model.load_state_dict(self.best_state_dict)
        test_loss, test_acc = self.evaluate_global_model(self.global_test_loader)
        self.logger.info(
            f"--final_global_test_loss : {test_loss:.4f} "
            f"--final_global_test_acc : {test_acc:.4f} "
            f"--selected_round : {self.best_round}\n"
        )
        record_server_result(
            {
                "phase": "final_test",
                "round": self.best_round,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "selected_round": self.best_round,
                "best_val_acc": self.best_val_acc,
                "best_val_loss": self.best_val_loss,
                "selected_for_test": 1,
            },
            self.args,
        )
        # server.pth should keep the current/last-round server model semantics.
        # Restore the last server state after testing the best checkpoint.
        self.model.load_state_dict(last_server_state)

    def get_client_train_size(self,client_id):
        # FedAvg 使用客户端训练样本数作为聚合权重。
        return get_client_train_size(self.args, client_id, meta=self.partition_meta)

    def aggregation_by_method(self):
        # 聚合器接口：
        # - fedavg：对完整 state_dict 按客户端训练样本数加权平均；
        # - expert_fedavg：普通层按客户端样本数聚合，专家层按每个 expert 实际处理样本数聚合。
        client_states = []
        client_sizes = []
        for id in self.clientsID_list:
            client_state_dict = torch.load(
                self.args.model_save_path + f"/{id}.pth",
                map_location="cpu",
            )
            client_states.append(client_state_dict)
            client_sizes.append(self.get_client_train_size(id))

        total_size = sum(client_sizes)
        if total_size <= 0:
            raise ValueError("FedAvg requires at least one training sample across clients")

        fedavg_state = self.aggregator.aggregate(
            client_updates=client_states,
            client_weights=client_sizes,
            global_model=self.model,
            expert_weights=getattr(self, "last_client_expert_usages", None),
        )
        self.model.load_state_dict(fedavg_state)
        self.logger.info(f"--aggregation_method : {self.args.agg_method}\n")
        self.logger.info(f"--client_train_sizes : {client_sizes}\n")

    def aggregation(self):
        self.aggregation_by_method()

