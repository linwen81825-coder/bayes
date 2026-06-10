import argparse
import logging
import os
import warnings

import torch.multiprocessing as mp

from configs import add_config_path_arguments, load_args
from data.data import CIFARPartitionBuilder
from fl.server import Server
from utils.utils import get_experiment_stem, resolve_device, set_seed

warnings.filterwarnings("ignore")


class ConsoleSummaryFilter(logging.Filter):
    """
    控制台日志过滤器。

    目标：
    - 控制台只显示训练摘要，例如 round / loss / final_acc / best_acc。
    - 其他完整 debug / diagnostics 日志不在控制台显示。
    - 文件日志不使用这个 filter，所以完整日志仍然会保存到 logs/*.log。
    """

    def filter(self, record):
        return bool(getattr(record, "to_console", False))


def configure_torch_multiprocessing():
    """
    配置 PyTorch 多进程共享策略。

    背景：
    DataLoader 在 num_workers > 0 时会使用多进程。
    PyTorch 默认的 file_descriptor 共享策略容易占用较多文件描述符，
    在 FL 多客户端反复创建/使用 DataLoader 的场景里，可能触发：
    OSError: [Errno 24] Too many open files

    file_system 策略可以降低文件描述符压力。
    这里放在训练入口最前面执行，确保创建 DataLoader / Server 前生效。
    """
    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        # 如果当前环境不允许重复设置或策略不可用，不中断训练。
        pass


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

    # 控制台日志：
    # 只显示带 extra={"to_console": True} 的精简摘要。
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(ConsoleSummaryFilter())
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # 文件日志：
    # 不加 filter，完整保留所有 info/debug/diagnostics 日志。
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{logger_name}.log"),
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def main():
    configure_torch_multiprocessing()

    cli_parser = argparse.ArgumentParser(description="Train with a YAML configuration file.")
    add_config_path_arguments(cli_parser)
    cli_args = cli_parser.parse_args()

    # Read experiment settings from the YAML config file under `configs/`.
    args = load_args(config_path=cli_args.config)
    args.device = resolve_device(args.device)

    set_seed(args.seed)

    logger = build_logger(args)

    logger.info(f"[Runtime] device={args.device}")

    if args.resume:
        logger.info("[Resume] Enabled. Skip rebuilding data partition and use existing partition files.")
    else:
        logger.info("[Partition] Rebuilding data partition before training...")
        CIFARPartitionBuilder(args=args).build()
        logger.info("[Partition] Data partition rebuilt successfully.")

    set_seed(args.seed)

    # 项目主入口：创建服务端对象，然后启动联邦训练流程。
    Server(args=args, logger=logger).train()


if __name__ == "__main__":
    main()