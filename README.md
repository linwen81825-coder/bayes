# Federated Learning Experiment

这是一个基于 CIFAR10/CIFAR100 的 FL + MoE 实验项目，当前模型分支为 `Hybrid CNN Stem + Switch Transformer`，默认聚合方法为 `expert_bayes_meta`。

## 环境准备

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate bayes_env
```

如果环境已经存在，直接激活即可：

```bash
conda activate bayes_env
```

## 配置方式

项目现在使用 YAML 作为唯一配置来源，不再依赖 `configs/args.py + argparse`。
配置加载使用真正的 `PyYAML` 解析，也就是 `yaml.safe_load(...)`，不再使用手写的逐行 flat parser。

配置分为三份：

- `configs/data.yaml`
  - 数据集、划分协议、随机种子等
- `configs/train.yaml`
  - 联邦训练轮数、设备、结果输出路径等
- `configs/model.yaml`
  - 模型结构和优化器超参数等

项目入口会调用 `configs/__init__.py` 中的 `load_args()`，将三份 YAML 合并成一个扁平的 `args` 对象，因此项目内部仍然继续使用 `args.xxx` 访问配置。

这三份 YAML 的顶层都必须是 key-value mapping；空文件会按空配置处理。
三份 YAML 配置文件会在启动时读取并合并。为避免歧义，顶层 key 必须全局唯一；如果出现重复 key，`load_args()` 会直接报错，而不是静默覆盖。
默认假设从项目根目录运行 `python -m data.data` 和 `python train.py`；除非显式传入绝对路径，否则会读取项目根目录下的 `configs/data.yaml`、`configs/train.yaml` 和 `configs/model.yaml`。

## 实验切换方式

- 切 CIFAR10 / CIFAR100：修改 `configs/data.yaml` 中的 `data_name`
- 改 `alpha`：修改 `configs/data.yaml` 中的 `alpha`
- 切聚合方法：修改 `configs/train.yaml` 中的 `agg_method`
- 切模型：修改 `configs/model.yaml` 中的 `model_type`
  - `hybrid_switch_transformer`：CNN stem + Transformer
  - `switch_transformer`：patch embedding + Transformer
  - `switch_transformer` 现在支持显式 `patch_size`；该字段只对标准 Switch 生效
  - `hybrid_switch_transformer` 仍然使用 `token_grid_size` 控制 token 网格
  - 结果文件名现在会区分 `model_type`；对 `switch_transformer` 还会进一步区分 `patch_size`
- 改完 YAML 后，如果变动涉及数据划分（例如 `data_name`、`alpha`、`num_clients`、`global_val_ratio`），先运行 `python -m data.data`，再运行 `python train.py`
- 如果只改训练或模型配置，且不影响数据划分，可以直接重新训练；否则先重建 partition
- 如果想切换另一套 YAML，也可以使用轻量命令行入口：

```bash
python -m data.data
python train.py

python -m data.data --data_cfg configs/exp/cifar10.yaml --train_cfg configs/exp/train_fedavg.yaml --model_cfg configs/model.yaml
python train.py --data_cfg configs/exp/cifar10.yaml --train_cfg configs/exp/train_fedavg.yaml --model_cfg configs/model.yaml
```

两种模型当前都提供一致的 MoE 辅助接口，包括 expert/router state dict 提取和 parameter groups。

## 运行顺序

先按需要修改上述 YAML 文件，再生成数据划分：

```bash
python -m data.data
```

再启动训练：

```bash
python train.py
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

如果训练时报原始 CIFAR 缺失，请检查 `configs/data.yaml` 中的 `data_path`，或者重新运行：

```bash
python -m data.data
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

如果需要读取 `best_server.pth`，请使用 `utils/utils.py` 中的 `load_best_server_checkpoint(path)`，不要把它当成纯 `state_dict` 直接使用。

## 常用命令

检查核心依赖版本：

```bash
python -c "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)"
```

检查 CUDA 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())"
```
