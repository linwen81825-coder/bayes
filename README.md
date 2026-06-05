# Federated Learning Experiment

这是一个基于 CIFAR10/CIFAR100 的 FL + MoE 实验项目，当前模型分支为 `Hybrid CNN Stem + Switch Transformer`，默认对非专家参数和专家参数都按客户端训练样本数加权聚合。

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

项目现在只使用一个配置文件：

- `configs/config.yaml`

`configs/config.yaml` 按功能分组：

- `data`：数据集和划分
- `federated`：联邦训练
- `runtime`：输出目录和断点续训
- `aggregation`：非专家/专家参数聚合
- `model`：模型主干
- `moe`：MoE 和 router

项目入口会调用 `configs/__init__.py` 中的 `load_args()`，将分组配置压平成一个扁平的 `args` 对象，因此项目内部仍然继续使用 `args.xxx` 访问配置。

配置文件顶层必须是按功能分组的 mapping；每个分组下的 key 会在启动时压平。为避免歧义，压平后的 key 必须全局唯一；如果出现重复 key，`load_args()` 会直接报错，而不是静默覆盖。
默认假设从项目根目录运行 `python train.py`；除非显式传入绝对路径，否则会读取项目根目录下的 `configs/config.yaml`。`resume=false` 时，`train.py` 会在训练前自动重建并覆盖数据划分文件。

## 实验切换方式

- 切 CIFAR10 / CIFAR100：修改 `configs/config.yaml` 中的 `data.data_name`
- 改 `alpha`：修改 `configs/config.yaml` 中的 `data.alpha`
- 改客户端数：修改 `configs/config.yaml` 中的 `federated.num_clients`
- 切聚合方法：修改 `configs/config.yaml` 中的 `aggregation.non_expert_agg_method` 和 `aggregation.expert_agg_method`
- 改 backbone：修改 `configs/config.yaml` 中的 `model.backbone_type`
- 改 expert 数：修改 `configs/config.yaml` 中的 `moe.num_experts`
- 改输出目录：修改 `configs/config.yaml` 中的 `runtime.save_root`
- 续训：修改 `configs/config.yaml` 中的 `runtime.resume` 和 `runtime.resume_checkpoint`
- 改完 YAML 后，直接运行 `python train.py`；`resume=false` 时训练入口会自动重建 partition
- `resume=false` 时会覆盖已有的 `partition_meta.pt` 和 `partition_stats.json`
- 如果想切换另一套 YAML，也可以使用轻量命令行入口：

```bash
python train.py

python train.py --config configs/config.yaml
```

## 聚合配置

`configs/config.yaml` 的 `aggregation` 分组中使用两条聚合配置链路：

```yaml
aggregation:
  non_expert_agg_method: sample_weighted
  expert_agg_method: sample_weighted
```

- `non_expert_agg_method` 控制非专家参数聚合
- `expert_agg_method` 控制专家参数聚合
- `sample_weighted`：按客户端训练样本数加权平均
- `uniform`：每个客户端等权平均
- 后续新增聚合方法时，在 `fl/aggregators.py` 注册即可

## 断点续训

从头训练时保持默认配置：

```yaml
runtime:
  resume: false
  resume_checkpoint: latest
```

启动训练：

```bash
python train.py
```

中断后续训时改为：

```yaml
runtime:
  resume: true
  resume_checkpoint: latest
```

再次运行：

```bash
python train.py
```

- `resume=false` 时会重新划分数据并覆盖旧 CSV
- `resume=true` 时不会重新划分数据，不会覆盖旧 CSV，会从 `latest.pth` 的下一轮继续
- checkpoint 固定每轮保存一次
- checkpoint 保存在 `save_root/model/checkpoints/`

## 运行顺序

按需要修改 `configs/config.yaml` 后，直接启动训练；`resume=false` 时，`train.py` 会在训练前自动重建并覆盖数据划分文件：

```bash
python train.py
```

训练时控制台会显示实验级 tqdm 动态总进度条：
- 总步数 = server_epochs × num_clients
- 每完成一个客户端本地训练，进度条更新并刷新一次
- 进度条会显示动态条形进度、已用时间、预计剩余时间和平均每个 client 耗时
- resume=true 时，进度条会从已完成 round 对应的位置继续
- 详细训练日志仍然写入 save_root/result/ 下的日志文件，进度条不会写入日志文件

最小提速相关默认项：
- `federated.device` 支持 `auto` / `cuda` / `cuda:0` / `cpu`
- `device: auto` 表示有 GPU 就用 GPU，没有 GPU 自动使用 CPU
- `data.num_workers=2`、`pin_memory=true` 可提升 GPU 训练时数据加载和拷贝效率
- `runtime.in_memory_client_updates=true` 会让客户端模型更新通过内存传递，避免每轮写 client pth 再读取；聚合结果不变
- 如果需要调试旧 pth 流程，可以设置 `runtime.in_memory_client_updates=false`

## 数据协议

当前项目使用的是 index-based partition 协议：

- official `train` 全部作为客户端训练池 `client_train_pool`
- `client_train_pool` 通过 Dirichlet non-IID 划分得到各客户端的 `client_train_indices`
- official `test` 完整保留给服务器做 `global_test`
- `resume=false` 时，`python train.py` 会在训练前自动重建并覆盖 `partition_meta.pt` 和 `partition_stats.json`
- `partition_meta.pt` 只保存索引和元信息，不保存原始图像数据
- `partition_stats.json` 保存各 split 的样本规模和类别统计

这意味着训练阶段仍然需要 `data_path` 下存在原始 CIFAR 数据文件。`data/loader.py` 会基于：

- raw CIFAR dataset
- saved indices
- split-specific transforms

动态构造 `Dataset` / `DataLoader`。

如果训练时报原始 CIFAR 缺失，请检查 `configs/config.yaml` 中的 `data.data_path`，然后重新运行 `python train.py`。

## 训练与评估协议

- client 只训练自己的 `client_train`
- server 每轮在 `global_test` 上评估当前全局模型
- best model 选择规则：
  - 先比较 `global_test_acc`
  - 若相同，再比较 `global_test_loss`
- 不再单独划分验证集

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
  - server 侧逐轮 `global_test` 结果
- `save/result/logs/*.log`
  - 本次实验的完整日志

CSV 和日志文件名都会包含：

- `data_name`
- `num_clients`
- `alpha`
- `seed`
- `non_expert_agg_method`
- `expert_agg_method`

## Checkpoint 约定

项目里模型文件有两种格式：

- `server.pth`
  - 服务端当前模型的纯 `state_dict`
- `save/model/{client_id}.pth`
  - 默认 `runtime.in_memory_client_updates=true` 时，不依赖每个客户端的 `{client_id}.pth` 做聚合
  - 只有 `runtime.in_memory_client_updates=false` 的 pth fallback 模式下，才会写出 `save/model/{client_id}.pth`
- `best_server.pth`
  - checkpoint dict，至少包含：
    - `model_state_dict`
    - `best_round`
    - `best_test_acc`
    - `best_test_loss`
- `save/model/checkpoints/round_*.pth` / `save/model/checkpoints/latest.pth`
  - 断点续训 checkpoint，包含服务端模型、已完成轮次、best 指标、best 模型、随机数状态和配置快照

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
