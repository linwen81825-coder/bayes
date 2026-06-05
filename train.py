import argparse
import logging
import os
import warnings

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


def main():
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
