import argparse

# 这个文件专门负责“命令行参数配置”。
# 入口文件会通过 `from configs.args import parse` 拿到同一个 ArgumentParser，
# 然后调用 `parse.parse_args()` 读取运行时传入的参数。
parse = argparse.ArgumentParser(description="system")

###################################### data settings ################################################
# 数据集相关参数：
# - data_name：选择使用哪个数据集
# - data_path：原始数据下载/存放位置
# - global_val_ratio：official train set 中留给 server-side global validation 的比例
# - data_save_path：划分后的索引协议和统计信息保存目录
# - alpha：Dirichlet 分布参数，越小表示客户端之间数据越不均匀（Non-IID 越强）
parse.add_argument("--data_name",type=str,default="cifar100",choices=['cifar10', 'cifar100'])
parse.add_argument("--data_path",type=str,default="./data")
parse.add_argument("--global_val_ratio",type=float,default=0.1)
parse.add_argument("--data_save_path",type=str,default="./save/data")
parse.add_argument("--batch_size",type=int,default=32)
parse.add_argument("--min_datasize",type=int,default=32)
parse.add_argument("--alpha",type=float,default=0.1)
parse.add_argument("--seed",type=int,default=1)
parse.add_argument("--partition_meta_name",type=str,default="partition_meta.pt")
parse.add_argument("--partition_stats_name",type=str,default="partition_stats.json")
parse.add_argument("--num_workers",type=int,default=0)
parse.add_argument("--pin_memory",action=argparse.BooleanOptionalAction,default=False)


###################################### base settings ################################################
# 联邦训练基础参数：
# - num_clients：客户端数量
# - server_epochs：服务端通信轮数，也就是联邦学习的全局轮数
# - client_epochs：每个客户端每轮本地训练多少个 epoch
# - device：训练设备，常见取值为 "cpu" 或 "cuda"
# - save_result：日志和 CSV 训练结果保存目录
# - agg_method：expert_fedavg 为专家级样本数加权聚合，fedavg 为完整模型标准 FedAvg
parse.add_argument("--num_clients",type=int,default=20)
parse.add_argument("--server_epochs",type=int,default=50)
parse.add_argument("--client_epochs",type=int,default=1)
parse.add_argument("--device",type=str,default="cuda")
parse.add_argument("--save_result",type=str,default="./save/result")
parse.add_argument("--agg_method",type=str,default="expert_fedavg",choices=["expert_fedavg","fedavg"])


###################################### model settings ################################################
# 模型相关参数：
# - num_experts：Switch FFN 中的专家数量
# - learning_rate：优化器学习率
# - model_save_path：服务端模型和客户端模型保存目录
parse.add_argument("--model_type",type=str,default="hybrid_switch_transformer",choices=["hybrid_switch_transformer"])
parse.add_argument("--num_experts",type=int,default=8)
parse.add_argument("--dropout",type=float,default=0.2)
parse.add_argument("--learning_rate",type=float,default=5e-4)
parse.add_argument("--model_save_path",type=str,default="./save/model")

# Hybrid CNN Stem + Switch Transformer 参数。
# num_layers 是 depth 的别名；如果显式传入 num_layers，会优先使用 num_layers。
parse.add_argument("--embed_dim",type=int,default=128)
parse.add_argument("--num_heads",type=int,default=4)
parse.add_argument("--mlp_ratio",type=float,default=4.0)
parse.add_argument("--depth",type=int,default=4)
parse.add_argument("--num_layers",type=int,default=None)
parse.add_argument("--moe_layers",type=str,default="1,3")
parse.add_argument("--top_k",type=int,default=1)
parse.add_argument("--router_aux_loss_coef",type=float,default=0.01)
parse.add_argument("--router_z_loss_coef",type=float,default=0.001)
parse.add_argument("--router_jitter_noise",type=float,default=0.0)
parse.add_argument("--capacity_factor",type=float,default=1.25)
parse.add_argument("--min_capacity",type=int,default=4)
parse.add_argument("--drop_tokens",action=argparse.BooleanOptionalAction,default=True)
parse.add_argument("--stem_channels",type=int,default=64)
parse.add_argument("--token_grid_size",type=int,default=8)
parse.add_argument("--use_cls_token",action=argparse.BooleanOptionalAction,default=False)
