"""Project configuration loader backed by PyYAML."""

from __future__ import annotations

import argparse
import re
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
    "agg_method",
    "bayes_sgld_steps",
    "bayes_sgld_burnin",
    "bayes_sgld_lr",
    "bayes_ai_max",
    "bayes_evidence_batches",
    "bayes_min_expert_tokens",
    "bayes_meta_steps",
    "bayes_meta_lr",
    "bayes_gamma0_init",
    "bayes_n0_init",
    "bayes_update_precision",
    "bayes_update_strength",
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
_DEFAULT_CONFIG_VALUES = {
    "save_root": "save",
    "allow_overwrite": False,
    "auto_prepare_data": True,
    "force_repartition": False,
    "use_tqdm": True,
    "progress_bar": True,
    "progress_bar_leave": False,
    "progress_bar_mininterval": 1.0,
    "progress_ncols": 140,
    "progress_log_interval": 1,
    "progress_force_tty": False,
    "progress_expert_bar": False,
}
_RUN_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


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


def _sanitize_run_name(run_name: object) -> str:
    cleaned = _RUN_NAME_PATTERN.sub("_", str(run_name).strip())
    cleaned = cleaned.strip("._-")
    if cleaned == "":
        raise ValueError(
            "`run_name` is empty after sanitization. "
            "Please set a non-empty run_name, for example `run_name: exp7`."
        )
    return cleaned


def _default_run_name_from_config_path(config_path: Path) -> str:
    if config_path.parent.name == "configs":
        return "default"
    return config_path.parent.name


def _as_bool(value: object, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"`{key}` must be a boolean value, got {value!r}.")


def _derive_output_paths(merged_config: dict, train_cfg_path: Path) -> None:
    for key, value in _DEFAULT_CONFIG_VALUES.items():
        merged_config.setdefault(key, value)

    run_name = merged_config.get("run_name")
    if run_name is None or str(run_name).strip() == "":
        run_name = _default_run_name_from_config_path(train_cfg_path)

    sanitized_run_name = _sanitize_run_name(run_name)
    save_root = Path(str(merged_config["save_root"]))
    run_dir = save_root / sanitized_run_name

    merged_config["run_name"] = sanitized_run_name
    for bool_key in ["allow_overwrite", "auto_prepare_data", "force_repartition"]:
        merged_config[bool_key] = _as_bool(merged_config[bool_key], bool_key)
    merged_config["run_dir"] = str(run_dir)
    merged_config["data_save_path"] = str(run_dir / "data")
    merged_config["model_save_path"] = str(run_dir / "model")
    merged_config["save_result"] = str(run_dir / "result")


def _is_non_empty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _check_output_overwrite(args: SimpleNamespace, output_phase: str | None) -> None:
    if output_phase is None or args.allow_overwrite:
        return

    phase_targets = {
        "data": [args.data_save_path],
        "train": [args.model_save_path, args.save_result],
        "all": [args.data_save_path, args.model_save_path, args.save_result],
    }
    if output_phase not in phase_targets:
        raise ValueError(
            f"Unsupported output_phase: {output_phase!r}. "
            "Expected one of: data, train, all."
        )

    blocked_paths = [
        path
        for path in phase_targets[output_phase]
        if _is_non_empty_dir(_resolve_config_path(path))
    ]
    if not blocked_paths:
        return

    formatted_paths = ", ".join(str(path) for path in blocked_paths)
    raise FileExistsError(
        "Output directory already exists and is not empty: "
        f"{formatted_paths}. To protect existing experiment outputs, "
        "please choose a new `run_name`, or explicitly set "
        "`allow_overwrite: true` if you really want to reuse this run."
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
    output_phase: str | None = None,
):
    """Load flat YAML config files and return an args-like namespace."""

    config_items = []
    resolved_train_cfg_path = _resolve_config_path(train_cfg_path)
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

    _derive_output_paths(merged_config, resolved_train_cfg_path)
    _raise_if_missing_required_keys(merged_config)

    args = SimpleNamespace(**merged_config)
    _check_output_overwrite(args, output_phase)
    return args


__all__ = [
    "DEFAULT_DATA_CFG_PATH",
    "DEFAULT_TRAIN_CFG_PATH",
    "DEFAULT_MODEL_CFG_PATH",
    "add_config_path_arguments",
    "load_args",
]
