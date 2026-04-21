"""Project configuration loader backed by PyYAML."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parent
DEFAULT_DATA_CFG_PATH = "configs/data.yaml"
DEFAULT_TRAIN_CFG_PATH = "configs/train.yaml"
DEFAULT_MODEL_CFG_PATH = "configs/model.yaml"
_REQUIRED_CONFIG_KEYS = (
    "data_name",
    "data_path",
    "global_val_ratio",
    "data_save_path",
    "batch_size",
    "min_datasize",
    "alpha",
    "seed",
    "partition_meta_name",
    "partition_stats_name",
    "num_workers",
    "pin_memory",
    "num_clients",
    "server_epochs",
    "client_epochs",
    "device",
    "save_result",
    "agg_method",
    "model_save_path",
    "model_type",
    "num_experts",
    "dropout",
    "learning_rate",
    "embed_dim",
    "num_heads",
    "mlp_ratio",
    "depth",
    "num_layers",
    "moe_layers",
    "top_k",
    "router_aux_loss_coef",
    "router_z_loss_coef",
    "router_jitter_noise",
    "capacity_factor",
    "min_capacity",
    "drop_tokens",
    "stem_channels",
    "token_grid_size",
    "use_cls_token",
)


def _resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


def _load_yaml_mapping(config_path: str | Path) -> dict:
    """Load one YAML file and require a top-level mapping."""

    config_path = _resolve_config_path(str(config_path))
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML syntax in config file: {config_path}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read config file: {config_path}") from exc

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {config_path}")

    return data


def _raise_if_duplicate_keys(named_configs: list[tuple[str, dict]]) -> None:
    duplicate_details = []
    for index, (left_name, left_cfg) in enumerate(named_configs):
        for right_name, right_cfg in named_configs[index + 1:]:
            duplicate_keys = sorted(set(left_cfg) & set(right_cfg))
            if duplicate_keys:
                duplicate_details.append(
                    f"{left_name} and {right_name}: {duplicate_keys}"
                )

    if duplicate_details:
        raise ValueError(
            "Duplicate config keys found across YAML files. "
            + "; ".join(duplicate_details)
            + ". Please keep keys unique across data.yaml, train.yaml, and model.yaml."
        )


def _raise_if_missing_required_keys(merged_config: dict) -> None:
    missing_keys = sorted(key for key in _REQUIRED_CONFIG_KEYS if key not in merged_config)
    if missing_keys:
        raise ValueError(
            f"Missing required config keys: {missing_keys}. "
            "Please check configs/data.yaml, configs/train.yaml, and configs/model.yaml."
        )


def add_config_path_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the lightweight YAML-path CLI overrides used by the entrypoints."""

    parser.add_argument("--data_cfg", default=DEFAULT_DATA_CFG_PATH)
    parser.add_argument("--train_cfg", default=DEFAULT_TRAIN_CFG_PATH)
    parser.add_argument("--model_cfg", default=DEFAULT_MODEL_CFG_PATH)
    return parser


def load_args(
    data_cfg_path: str = DEFAULT_DATA_CFG_PATH,
    train_cfg_path: str = DEFAULT_TRAIN_CFG_PATH,
    model_cfg_path: str = DEFAULT_MODEL_CFG_PATH,
):
    """Load flat YAML config files and return an args-like namespace."""

    config_items = []
    for config_path_str in [data_cfg_path, train_cfg_path, model_cfg_path]:
        try:
            resolved_path = _resolve_config_path(config_path_str)
            config_items.append((resolved_path.name, _load_yaml_mapping(resolved_path)))
        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as exc:
            config_path = _resolve_config_path(config_path_str)
            raise RuntimeError(f"Failed to load YAML config from `{config_path}`.") from exc

    _raise_if_duplicate_keys(config_items)

    merged_config = {}
    for _, config in config_items:
        merged_config.update(config)

    _raise_if_missing_required_keys(merged_config)

    return SimpleNamespace(**merged_config)


__all__ = [
    "DEFAULT_DATA_CFG_PATH",
    "DEFAULT_TRAIN_CFG_PATH",
    "DEFAULT_MODEL_CFG_PATH",
    "add_config_path_arguments",
    "load_args",
]
