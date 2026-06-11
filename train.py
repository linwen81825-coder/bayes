import os
import sys
from pathlib import Path


def _read_bootstrap_config_values():
    # 在导入任何可能触发 torch/CUDA 的项目模块前，只读 YAML 里的 seed 和 CUBLAS 配置。
    config_path = "configs/config.yaml"
    argv = sys.argv[1:]
    for idx, item in enumerate(argv):
        if item == "--config" and idx + 1 < len(argv):
            config_path = argv[idx + 1]
            break
        if item.startswith("--config="):
            config_path = item.split("=", 1)[1]
            break

    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    try:
        import yaml

        with path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file) or {}
    except Exception:
        raw_config = {}

    flattened = {}
    if isinstance(raw_config, dict):
        for group_config in raw_config.values():
            if isinstance(group_config, dict):
                flattened.update(group_config)

    return {
        "seed": flattened.get("seed", 0),
        "cublas_workspace_config": flattened.get(
            "cublas_workspace_config",
            ":4096:8",
        ),
    }


_BOOTSTRAP_CONFIG = _read_bootstrap_config_values()
# CUBLAS_WORKSPACE_CONFIG 必须在 CUDA 上下文创建前设置，放在项目模块和 torch 导入之前。
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    str(_BOOTSTRAP_CONFIG["cublas_workspace_config"]),
)
# PYTHONHASHSEED 在解释器启动时最有效；这里尽早跟随配置 seed，并在读取 args 后再次同步。
os.environ.setdefault("PYTHONHASHSEED", str(_BOOTSTRAP_CONFIG["seed"]))

import argparse
import logging
import warnings

import torch

from configs import add_config_path_arguments, load_args
from data.data import CIFARPartitionBuilder
from fl.server import Server
from utils.utils import get_experiment_stem, resolve_device, set_seed

warnings.filterwarnings("ignore")


def build_logger(args):
    log_dir = os.path.join(args.save_result, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件名中加入关键实验配置，方便区分不同实验结果。
    logger_name = get_experiment_stem(args)

    # Python 标准 logging 用法：
    # logger 负责统一接收日志，handler 决定日志输出到哪里。
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    # 控制台日志：训练时直接在终端输出。
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    # 文件日志：同时把训练过程保存到 save/result/logs/*.log。
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{logger_name}.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def log_determinism_state(logger, args, cublas_config_warning):
    # 启动时只打印一次确定性状态，方便复现实验时核对运行环境。
    deterministic_algorithms = (
        torch.are_deterministic_algorithms_enabled()
        if hasattr(torch, "are_deterministic_algorithms_enabled")
        else "unavailable"
    )
    if cublas_config_warning:
        logger.warning(
            "[Determinism] CUBLAS_WORKSPACE_CONFIG 与配置不一致；"
            "CUDA 上下文创建后再修改可能无效，请在启动命令环境中确认。"
        )
    logger.info(
        "[Determinism] "
        f"seed={args.seed} "
        f"deterministic={bool(getattr(args, 'deterministic', True))} "
        f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')} "
        f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} "
        f"cudnn.deterministic={torch.backends.cudnn.deterministic} "
        f"cudnn.benchmark={torch.backends.cudnn.benchmark} "
        f"deterministic_algorithms={deterministic_algorithms} "
        f"num_workers={getattr(args, 'num_workers', None)}"
    )


def main():
    cli_parser = argparse.ArgumentParser(description="Train with a YAML configuration file.")
    add_config_path_arguments(cli_parser)
    cli_args = cli_parser.parse_args()

    # Read experiment settings from the YAML config file under `configs/`.
    args = load_args(config_path=cli_args.config)
    args.deterministic = bool(getattr(args, "deterministic", True))
    args.deterministic_warn_only = bool(getattr(args, "deterministic_warn_only", True))
    args.cublas_workspace_config = str(
        getattr(args, "cublas_workspace_config", ":4096:8")
    )
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    cublas_config_warning = (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") != args.cublas_workspace_config
    )
    args.device = resolve_device(args.device)
    set_seed(
        args.seed,
        deterministic=args.deterministic,
        deterministic_warn_only=args.deterministic_warn_only,
    )
    logger = build_logger(args)
    logger.info(f"[Runtime] device={args.device}")
    log_determinism_state(logger, args, cublas_config_warning)

    if args.resume:
        logger.info("[Resume] Enabled. Skip rebuilding data partition and use existing partition files.")
    else:
        logger.info("[Partition] Rebuilding data partition before training...")
        CIFARPartitionBuilder(args=args).build()
        logger.info("[Partition] Data partition rebuilt successfully.")

    set_seed(
        args.seed,
        deterministic=args.deterministic,
        deterministic_warn_only=args.deterministic_warn_only,
    )
    # 项目主入口：创建服务端对象，然后启动联邦训练流程。
    Server(args=args, logger=logger).train()


if __name__ == "__main__":
    main()
