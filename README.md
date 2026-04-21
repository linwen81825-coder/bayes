# Federated Learning Experiment

这是一个基于 CIFAR10/CIFAR100 的 FL + MoE 实验项目，当前模型分支为 `Hybrid CNN Stem + Switch Transformer`，默认聚合方法为 `expert_fedavg`。

## 环境准备

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate fedwolf
```

如果环境已经存在，直接激活即可：

```bash
conda activate fedwolf
```

## 运行顺序

先生成数据划分：

```bash
python -m data.data --data_name cifar10 --num_clients 2
```

再启动训练：

```bash
python train.py --data_name cifar10 --num_clients 2 --server_epochs 1 --client_epochs 1 --device cpu
```

## 数据协议

当前项目使用的是 index-based partition 协议：

- official `train` 先做分层切分，得到 `global_val` 和 `federated_train_pool`
- `federated_train_pool` 再通过 Dirichlet non-IID 划分得到各客户端的 `client_train_indices`
- official `test` 直接作为 `global_test`
- `partition_meta.pt` 只保存索引和元信息，不保存原始图像数据
- `partition_stats.json` 保存各 split 的样本规模和类别统计

这意味着训练阶段仍然需要 `data_path` 下存在原始 CIFAR 数据文件。`data/loader.py` 会基于：

- raw CIFAR dataset
- saved indices
- split-specific transforms

动态构造 `Dataset` / `DataLoader`。

如果训练时报原始 CIFAR 缺失，请检查 `--data_path`，或者重新运行：

```bash
python -m data.data --data_name cifar10 --num_clients 2
```

## 训练与评估协议

- client 只训练自己的 `client_train`
- server 每轮在 `global_val` 上评估当前全局模型
- best model 选择规则：
  - 先比较 `global_val_acc`
  - 若相同，再比较 `global_val_loss`
- `global_test` 不参与模型选择，只在训练结束后做最终评估

## 输出文件

### 数据划分

- `save/data/partition_meta.pt`
  - 索引划分协议和元信息
- `save/data/partition_stats.json`
  - 各 split 的样本数量和类别分布统计

### 模型文件

- `save/model/server.pth`
  - 当前轮 / 最后一轮服务端模型的纯 `state_dict`
- `save/model/best_server.pth`
  - 带元信息的 checkpoint dict，不是纯 `state_dict`
- `save/model/{client_id}.pth`
  - 每个客户端当前模型的纯 `state_dict`

### 结果与日志

- `save/result/detail/*.csv`
  - client 侧逐轮训练明细
- `save/result/server/*.csv`
  - server 侧逐轮 `global_val` 与最终 `global_test` 结果
- `save/result/logs/*.log`
  - 本次实验的完整日志

CSV 和日志文件名都会包含：

- `data_name`
- `num_clients`
- `alpha`
- `seed`
- `global_val_ratio`
- `agg_method`

## Checkpoint 约定

项目里模型文件有两种格式：

- `server.pth` / `save/model/{client_id}.pth`
  - 纯模型参数，直接保存 `state_dict`
- `best_server.pth`
  - checkpoint dict，至少包含：
    - `model_state_dict`
    - `best_round`
    - `best_val_acc`
    - `best_val_loss`

如果需要读取 `best_server.pth`，请使用 [utils/utils.py](/home/cjq/Project/FedWolf/utils/utils.py) 中的 `load_best_server_checkpoint(path)`，不要把它当成纯 `state_dict` 直接使用。

## 常用命令

检查核心依赖版本：

```bash
python -c "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)"
```

检查 CUDA 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())"
```
