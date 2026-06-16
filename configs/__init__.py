"""Project configuration loader backed by PyYAML."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parent
DEFAULT_CONFIG_PATH = "configs/config.yaml"
_OPTIONAL_CONFIG_DEFAULTS = {
    "use_server_meta_validation": False,
    "server_meta_validation_size": 1000,
    "server_meta_validation_balanced": True,
    "server_meta_validation_seed_offset": 9100,
    "uoc_foga_pism_objective": "meta_validation_loss",
    "uoc_foga_pism_input_features": [
        "client_loss",
        "expert_activation_frequency",
    ],
    "uoc_foga_pism_meta_loss_max_batches": 1,
    "uoc_foga_pism_recompute_weights_after_meta_step": True,
}

_REQUIRED_CONFIG_KEYS = (
    "data_name",
    "data_path",
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
    "save_root",
    "non_expert_agg_method",
    "expert_agg_method",
    "resume",
    "resume_checkpoint",
    "in_memory_client_updates",
    "model_type",
    "backbone_type",
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


def _flatten_grouped_config(config: dict) -> dict:
    flattened = {}
    for group_name, group_config in config.items():
        if not isinstance(group_config, dict):
            raise ValueError(
                f"Top-level config section `{group_name}` must be a mapping."
            )
        for key, value in group_config.items():
            if key in flattened:
                raise ValueError(
                    f"Duplicate config key after flatten: {key!r}. "
                    f"Please keep keys unique across config sections."
                )
            flattened[key] = value
    return flattened


def _raise_if_missing_required_keys(merged_config: dict) -> None:
    missing_keys = sorted(key for key in _REQUIRED_CONFIG_KEYS if key not in merged_config)
    if missing_keys:
        raise ValueError(
            f"Missing required config keys: {missing_keys}. "
            "Please check configs/config.yaml."
        )


def _validate_aggregation_methods(merged_config: dict) -> None:
    non_expert_methods = {"sample_weighted", "uniform"}
    expert_methods = {"sample_weighted", "uniform", "uoc_foga_expert_align", "uoc_foga_pism_expert_align"}

    non_expert_method = merged_config.get("non_expert_agg_method")
    if non_expert_method not in non_expert_methods:
        raise ValueError(
            "non_expert_agg_method must be one of "
            f"{sorted(non_expert_methods)}, got {non_expert_method!r}"
        )

    expert_method = merged_config.get("expert_agg_method")
    if expert_method not in expert_methods:
        raise ValueError(
            "expert_agg_method must be one of "
            f"{sorted(expert_methods)}, got {expert_method!r}"
        )


def _validate_device_config(merged_config: dict) -> None:
    device = str(merged_config["device"]).strip().lower()
    is_cuda_device = device == "cuda" or (
        device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
    )
    if device not in {"auto", "cpu"} and not is_cuda_device:
        raise ValueError("device must be one of: auto, cpu, cuda, cuda:<index>")


def _validate_checkpoint_config(merged_config: dict) -> None:
    resume = merged_config["resume"]
    if not isinstance(resume, bool):
        raise ValueError("resume must be a boolean: true or false")

    resume_checkpoint = merged_config["resume_checkpoint"]
    if not isinstance(resume_checkpoint, str) or not resume_checkpoint:
        raise ValueError("resume_checkpoint must be a non-empty string")


def _validate_runtime_config(merged_config: dict) -> None:
    if not isinstance(merged_config["in_memory_client_updates"], bool):
        raise ValueError("in_memory_client_updates must be a boolean")


def _derive_save_paths(merged_config: dict) -> None:
    save_root = Path(str(merged_config["save_root"]))
    merged_config["data_save_path"] = str(save_root / "data")
    merged_config["save_result"] = str(save_root / "result")
    merged_config["model_save_path"] = str(save_root / "model")


def add_config_path_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the YAML config path CLI argument used by the entrypoints."""

    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    return parser


def load_args(config_path: str = DEFAULT_CONFIG_PATH):
    """Load grouped YAML config and return an args-like namespace."""

    raw_config = _load_yaml_mapping(config_path)
    merged_config = _flatten_grouped_config(raw_config)
    for key, value in _OPTIONAL_CONFIG_DEFAULTS.items():
        merged_config.setdefault(key, value)

    _raise_if_missing_required_keys(merged_config)
    _validate_aggregation_methods(merged_config)
    _validate_device_config(merged_config)
    _validate_checkpoint_config(merged_config)
    _validate_runtime_config(merged_config)
    _derive_save_paths(merged_config)

    return SimpleNamespace(**merged_config)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "add_config_path_arguments",
    "load_args",
]
