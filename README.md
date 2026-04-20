# Federated Learning Experiment

这是一个联邦学习实验项目，当前只保留 `Hybrid CNN Stem + Switch Transformer` 模型分支。默认聚合方法是专家级样本数加权 FedAvg。

## 环境准备

推荐使用 Conda 根据 `environment.yml` 创建环境：

```bash
conda env create -f environment.yml
conda activate fedwolf
```

当前推荐环境记录如下：

- Python: `3.10`
- PyTorch: `torch==2.11.0+cu128`
- TorchVision: `torchvision==0.26.0+cu128`
- NumPy: `numpy==2.2.6`
- CUDA wheel 源：`https://download.pytorch.org/whl/cu128`

如果环境已经存在，可以直接激活：

```bash
conda activate fedwolf
```

## 项目入口

项目有两个主要入口：

- `python -m data.data`：下载/加载 CIFAR10/CIFAR100，并生成 index-based 联邦划分协议。
- `train.py`：启动联邦学习训练流程。

因此运行顺序是：先生成划分协议，再运行 `train.py` 开始训练。

数据划分逻辑：

- 官方 train set 先按类别分层切出 `global_val`，剩余部分作为 `federated_train_pool`。
- `federated_train_pool` 再通过 Dirichlet non-IID 划分到各客户端，客户端只有 train split。
- 不保存 transform 后的样本 list，只保存 `partition_meta.pt` 中的索引和元信息。
- `partition_stats.json` 记录各 split 大小和类别分布。
- `data/loader.py` 会基于 raw CIFAR dataset + indices + transform 动态构造 DataLoader。
- 客户端使用 `train_transform`，服务端 `global_val` / `global_test` 使用 `eval_transform`。
- 服务器每轮用 `global_val` 做模型选择，训练结束后只用 `global_test` 做最终测试。

## 基本运行顺序

先生成客户端数据：

```bash
python -m data.data --data_name cifar10 --num_clients 2
```

再启动一次最小训练：

```bash
python train.py --data_name cifar10 --num_clients 2 --server_epochs 1 --client_epochs 1 --device cpu
```

## 模型选择

默认模型是 Hybrid CNN Stem + Switch Transformer：

```bash
python train.py --model_type hybrid_switch_transformer --agg_method expert_fedavg
```

启动训练时可以显式写出，也可以省略这两个默认参数：

```bash
python train.py --model_type hybrid_switch_transformer --data_name cifar10 --num_clients 2 --server_epochs 1 --client_epochs 1 --device cpu
```

这个分支的数据流是：浅层 CNN stem -> feature map tokenization -> Transformer blocks -> 部分 block 的 FFN 替换为 token-level top-1 Switch FFN -> token mean pooling -> classifier。默认 `expert_fedavg` 聚合会对普通层使用客户端训练样本数加权，对 `blocks.{layer}.ffn.experts.{id}` 专家参数使用该层该专家在对应客户端实际处理的 token 数加权；如果需要完整模型标准 FedAvg，可显式传入 `--agg_method fedavg`。

可以通过 `--depth` / `--num_layers`、`--moe_layers`、`--embed_dim`、`--num_heads`、`--mlp_ratio`、`--router_aux_loss_coef`、`--router_z_loss_coef`、`--router_jitter_noise` 等参数控制结构和路由损失。

Switch FFN 支持 expert capacity 控制，相关参数包括 `--capacity_factor`、`--min_capacity` 和 `--drop_tokens` / `--no-drop_tokens`。超出 capacity 的 token 默认不进入 expert，而是在 Transformer 残差路径中走 identity bypass。模型默认使用 token mean pooling，也可以用 `--use_cls_token` 启用 cls token 分类。

## 输出目录

项目运行时会使用以下目录：

- `save/data`：`partition_meta.pt` 和 `partition_stats.json`。
- `save/model`：服务端模型 `server.pth` 和客户端模型 `{client_id}.pth`。
- `save/result/detail`：训练过程的 CSV 明细结果。
- `save/result/logs`：训练日志文件。

这些目录会在数据生成和训练时自动创建。

## 常用命令

检查环境中的核心依赖版本：

```bash
python -c "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)"
```

检查 CUDA 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())"
```
