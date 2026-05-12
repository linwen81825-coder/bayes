import argparse
import logging
import os
import warnings

from configs import add_config_path_arguments, load_args
from data.data import ensure_partition_ready
from fl.server import Server
from utils.utils import get_experiment_stem, set_seed, should_use_tqdm

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

    # 文件日志：同时把训练过程保存到 save/{run_name}/result/logs/*.log。
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{logger_name}.log"), mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def main():
    cli_parser = argparse.ArgumentParser(description="Train with YAML configuration files.")
    add_config_path_arguments(cli_parser)
    cli_args = cli_parser.parse_args()

    # Read experiment settings from YAML config files under `configs/`.
    args = load_args(
        data_cfg_path=cli_args.data_cfg,
        train_cfg_path=cli_args.train_cfg,
        model_cfg_path=cli_args.model_cfg,
        output_phase="train",
    )
    set_seed(args.seed)
    logger = build_logger(args)
    ensure_partition_ready(args, logger=logger)
    # Re-seed after optional partition generation so model init/training randomness stays stable.
    set_seed(args.seed)

    # 项目主入口：创建服务端对象，然后启动联邦训练流程。
    try:
        from tqdm.contrib.logging import logging_redirect_tqdm
    except Exception:
        logging_redirect_tqdm = None

    server = Server(args=args, logger=logger)
    if logging_redirect_tqdm is not None and should_use_tqdm(args):
        with logging_redirect_tqdm():
            server.train()
    else:
        server.train()


if __name__ == "__main__":
    main()
