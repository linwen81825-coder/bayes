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
  - 联邦训练轮数、设备、实验名和覆盖保护等
- `configs/model.yaml`
  - 模型结构和优化器超参数等

项目入口会调用 `configs/__init__.py` 中的 `load_args()`，将三份 YAML 合并成一个扁平的 `args` 对象，因此项目内部仍然继续使用 `args.xxx` 访问配置。

输出目录现在由 `configs/train.yaml` 中的 `save_root` 和 `run_name` 自动派生：

- `args.data_save_path = {save_root}/{run_name}/data`
- `args.model_save_path = {save_root}/{run_name}/model`
- `args.save_result = {save_root}/{run_name}/result`

也就是说，开新实验时优先只改 `run_name`，不再手动维护三条输出路径。
如果旧配置没有显式填写 `run_name`，加载器会临时使用 `train.yaml` 所在配置目录名作为实验名；仍然建议新实验显式填写 `run_name`。

这三份 YAML 的顶层都必须是 key-value mapping；空文件会按空配置处理。
三份 YAML 配置文件会在启动时读取并合并。为避免歧义，顶层 key 必须全局唯一；如果出现重复 key，`load_args()` 会直接报错，而不是静默覆盖。
默认假设从项目根目录运行 `python train.py`；除非显式传入绝对路径，否则会读取项目根目录下的 `configs/data.yaml`、`configs/train.yaml` 和 `configs/model.yaml`。

## 实验切换方式

- 切 CIFAR10 / CIFAR100：修改 `configs/data.yaml` 中的 `data_name`
- 改 `alpha`：修改 `configs/data.yaml` 中的 `alpha`
- 切聚合方法：修改 `configs/train.yaml` 中的 `agg_method`
- 开新实验：修改 `configs/train.yaml` 中的 `run_name`
- 故意覆盖旧实验：保持同一个 `run_name`，并显式设置 `allow_overwrite: true`
- 防止误覆盖：保持默认 `allow_overwrite: false`
- 切模型：修改 `configs/model.yaml` 中的 `model_type`
  - `hybrid_switch_transformer`：CNN stem + Transformer
  - `switch_transformer`：patch embedding + Transformer
  - `switch_transformer` 现在支持显式 `patch_size`；该字段只对标准 Switch 生效
  - `hybrid_switch_transformer` 仍然使用 `token_grid_size` 控制 token 网格
  - 结果文件名现在会区分 `model_type`；对 `switch_transformer` 还会进一步区分 `patch_size`
- 默认 `auto_prepare_data: true`，运行 `python train.py` 时会自动检查并准备数据划分
- 如果 partition 不存在，训练入口会自动生成；如果 partition 已存在且匹配当前 YAML，会直接复用
- 如果 partition 已存在但与当前 YAML 不匹配，会默认报错，提示更换 `run_name` 或设置 `allow_overwrite: true` / `force_repartition: true`
- 如果想保留旧的手动方式，仍然可以先运行 `python -m data.data`
- 如果想切换另一套 YAML，也可以使用轻量命令行入口：

```bash
python train.py

python train.py --data_cfg configs/exp/cifar10.yaml --train_cfg configs/exp/train_fedavg.yaml --model_cfg configs/model.yaml
```

两种模型当前都提供一致的 MoE 辅助接口，包括 expert/router state dict 提取和 parameter groups。

## 运行顺序

先按需要修改上述 YAML 文件，然后直接启动训练：

```bash
python train.py
```

默认 `auto_prepare_data: true`，训练入口会自动检查：

- `{save_root}/{run_name}/data/partition_meta.pt`
- `{save_root}/{run_name}/data/partition_stats.json`

行为如下：

- 如果不存在：自动生成
- 如果存在且匹配当前 YAML：直接复用
- 如果存在但与当前 YAML 不匹配：默认报错，防止误覆盖
- 如果确实想覆盖旧 partition：设置 `allow_overwrite: true` 或 `force_repartition: true`
- 如果想安全续跑同一个实验：设置 `resume: true`，训练会从 `save/{run_name}/model/resume_checkpoint.pth` 继续，日志和 CSV 都会追加写入

旧的手动生成 partition 命令仍然保留：

```bash
python -m data.data
```

## 数据协议

当前项目使用的是 index-based partition 协议：

- official `train` 通过 Dirichlet non-IID 划分得到各客户端的 `client_train_indices`
- official `test` 保持为统一的 `global_test`，只由服务端评估
- `partition_meta.pt` 只保存索引和元信息，不保存原始图像数据
- `partition_stats.json` 保存各 split 的样本规模和类别统计

这意味着训练阶段仍然需要 `data_path` 下存在原始 CIFAR 数据文件。`data/loader.py` 会基于：

- raw CIFAR dataset
- saved indices
- split-specific transforms

动态构造 `Dataset` / `DataLoader`。

如果训练时报原始 CIFAR 缺失，请检查 `configs/data.yaml` 中的 `data_path`。默认情况下可以直接重新运行 `python train.py` 触发自动数据准备；也可以使用旧的手动命令：

```bash
python -m data.data
```

## 训练与评估协议

- client 只训练自己的 `client_train`
- server 每轮在官方 `global_test` 上评估当前全局模型
- 当前流程不再单独划分 `global_val`
- 当前 CSV 记录的是每轮 test loss / test acc，不做 best checkpoint 选择

## 输出文件

如果 `save_root: save` 且 `run_name: exp7`，输出目录结构为：

- `save/exp7/data`
- `save/exp7/model`
- `save/exp7/result`

当 `allow_overwrite: false` 时，如果对应阶段的目标目录已经非空，程序会在真正写入前直接报错，提示更换 `run_name` 或显式设置 `allow_overwrite: true`。

### 数据划分

- `save/{run_name}/data/partition_meta.pt`
  - 索引划分协议和元信息
- `save/{run_name}/data/partition_stats.json`
  - 各 split 的样本数量和类别分布统计

### 模型文件

- `save/{run_name}/model/server.pth`
  - 当前轮 / 最后一轮服务端模型的纯 `state_dict`
- `save/{run_name}/model/resume_checkpoint.pth`
  - 安全续跑用的完整训练状态，包含已完成轮数、服务端模型和 `expert_bayes_meta` 的 Bayes 状态
- `save/{run_name}/model/server_bayes_state.pth`
  - `expert_bayes_meta` 使用的服务端贝叶斯状态
- `save/{run_name}/model/{client_id}.pth`
  - 每个客户端当前模型的纯 `state_dict`

### 结果与日志

- `save/{run_name}/result/detail/*.csv`
  - client 侧逐轮训练明细
- `save/{run_name}/result/server/*.csv`
  - server 侧逐轮 `global_test` 结果
- `save/{run_name}/result/logs/*.log`
  - 本次实验的完整日志

CSV 和日志文件名都会包含：

- `data_name`
- `num_clients`
- `alpha`
- `seed`
- `agg_method`
- `run_name`

## Checkpoint 约定

项目里模型文件有两种格式：

- `server.pth` / `save/{run_name}/model/{client_id}.pth`
  - 纯模型参数，直接保存 `state_dict`
- `server_bayes_state.pth`
  - `expert_bayes_meta` 的全局贝叶斯状态，不是模型 `state_dict`

## 安全续跑

如果实验中断，推荐直接在原 `run_name` 下继续：

```yaml
run_name: exp1
server_epochs: 80
resume: true
resume_checkpoint_path: null
allow_overwrite: false
```

然后重新运行：

```bash
python train.py
```

续跑时：

- 日志文件会追加，不会覆盖
- detail CSV 和 server CSV 会追加，不会重写表头
- `round` 会接着之前的完成轮数继续
- `expert_bayes_meta` 会同时恢复 `server_bayes_state.pth`
- `resume_allow_legacy_checkpoint: true` 时，如果没有 `resume_checkpoint.pth`，会尝试用 `server.pth` + server CSV 做兼容续跑

## 常用命令

检查核心依赖版本：

```bash
python -c "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)"
```

检查 CUDA 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())"
```
