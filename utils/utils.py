import csv
import os
import random

import numpy as np
import torch


def set_seed(seed:int, deterministic=True, deterministic_warn_only=True):
    """Set Python, NumPy, Torch, and CUDA seeds from one project-level value."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # 复现实验用的强确定性设置，可能降低 cuDNN/CUDA 算子速度。
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(
                    True,
                    warn_only=bool(deterministic_warn_only),
                )
            except TypeError:
                torch.use_deterministic_algorithms(True)


def resolve_device(device: str) -> str:
    """解析训练设备。auto 表示优先使用 GPU，没有 GPU 时回退 CPU。"""
    device = str(device).strip().lower()

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        return "cpu"

    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            return "cpu"

        index_text = device.split(":", 1)[1]
        try:
            index = int(index_text)
        except ValueError as exc:
            raise ValueError(f"Unsupported CUDA device index: {index_text!r}") from exc

        device_count = torch.cuda.device_count()
        if index < 0 or index >= device_count:
            raise ValueError(
                f"CUDA device index {index} is unavailable; "
                f"current available GPU count is {device_count}."
            )
        return device

    raise ValueError(f"Unsupported device: {device!r}")


def capture_rng_state():
    """保存当前随机数状态，用于断点续训后尽量保持实验连续性。"""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_state):
    """恢复 checkpoint 中保存的随机数状态。"""
    if not rng_state:
        return

    if "python" in rng_state:
        random.setstate(rng_state["python"])

    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])

    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])

    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def get_experiment_stem(args):
    return (
        f"data_{args.data_name}_"
        f"clients_{args.num_clients}_"
        f"alpha_{args.alpha}_"
        f"seed_{args.seed}_"
        f"non_expert_agg_{args.non_expert_agg_method}_"
        f"expert_agg_{args.expert_agg_method}"
    )


def get_csv_path(args):
    # 读取参数后，拼出本次实验的 CSV 结果文件路径。
    # CSV 文件会记录每一轮、每个客户端的 loss/acc。
    detail_dir = os.path.join(args.save_result, "detail")
    filename = f"{get_experiment_stem(args)}.csv"
    return os.path.join(detail_dir, filename)


def get_server_csv_path(args):
    server_dir = os.path.join(args.save_result, "server")
    filename = f"{get_experiment_stem(args)}.csv"
    return os.path.join(server_dir, filename)


def load_best_server_checkpoint(path):
    """Load `best_server.pth`, which is a checkpoint dict, not a pure state_dict.

    By convention:
    - `server.pth` and client `{id}.pth` store pure model state_dict objects.
    - `best_server.pth` stores a checkpoint dict with model_state_dict plus best metrics.
    """

    checkpoint = torch.load(path, map_location="cpu")
    required_keys = {
        "model_state_dict",
        "best_round",
        "best_test_acc",
        "best_test_loss",
    }

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"`{path}` must be a checkpoint dict for best_server.pth, got {type(checkpoint).__name__}."
        )

    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(
            f"`{path}` is missing checkpoint keys {sorted(missing_keys)}. "
            "best_server.pth must contain model_state_dict, best_round, best_test_acc, and best_test_loss."
        )

    return checkpoint

def init_result_csv(args, overwrite=True):
    """初始化结果 CSV，写入表头。

    Server 初始化时会调用一次；断点续训时保留已有 CSV 并继续追加。
    """

    csv_path = get_csv_path(args)
    if not overwrite and os.path.exists(csv_path):
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['T', 'client_epoch', 'client_id',"train_loss","train_acc","router_aux_loss","router_z_loss"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def init_server_result_csv(args, overwrite=True):
    csv_path = get_server_csv_path(args)
    if not overwrite and os.path.exists(csv_path):
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'phase',
            'round',
            'test_loss',
            'test_acc',
            'best_test_acc',
            'best_test_loss',
            'is_best',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def record_result(record_dic:dict, args):
    """追加写入一条客户端训练记录。"""

    csv_path = get_csv_path(args)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['T', 'client_epoch', 'client_id', "train_loss", "train_acc", "router_aux_loss", "router_z_loss"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(record_dic)


def record_server_result(record_dic:dict, args):
    csv_path = get_server_csv_path(args)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = [
            'phase',
            'round',
            'test_loss',
            'test_acc',
            'best_test_acc',
            'best_test_loss',
            'is_best',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(record_dic)
