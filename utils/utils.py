import csv
import os
import random

import numpy as np
import torch


def set_seed(seed:int):
    """Set Python, NumPy, Torch, and CUDA seeds from one project-level value."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_experiment_stem(args):
    return (
        f"data_{args.data_name}_"
        f"clients_{args.num_clients}_"
        f"alpha_{args.alpha}_"
        f"seed_{args.seed}_"
        f"gval_{args.global_val_ratio}_"
        f"agg_{args.agg_method}"
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
        "best_val_acc",
        "best_val_loss",
    }

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"`{path}` must be a checkpoint dict for best_server.pth, got {type(checkpoint).__name__}."
        )

    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(
            f"`{path}` is missing checkpoint keys {sorted(missing_keys)}. "
            "best_server.pth must contain model_state_dict, best_round, best_val_acc, and best_val_loss."
        )

    return checkpoint

def init_result_csv(args):
    """初始化结果 CSV，写入表头。

    Server 初始化时会调用一次，所以每次重新运行训练会覆盖同名 CSV。
    """

    csv_path = get_csv_path(args)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['T', 'client_epoch', 'client_id',"train_loss","train_acc","router_aux_loss","router_z_loss"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def init_server_result_csv(args):
    csv_path = get_server_csv_path(args)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'phase',
            'round',
            'val_loss',
            'val_acc',
            'best_val_acc',
            'best_val_loss',
            'is_best',
            'selected_for_test',
            'test_loss',
            'test_acc',
            'selected_round',
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
            'val_loss',
            'val_acc',
            'best_val_acc',
            'best_val_loss',
            'is_best',
            'selected_for_test',
            'test_loss',
            'test_acc',
            'selected_round',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(record_dic)
