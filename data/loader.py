import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


EXPECTED_PROTOCOL = "server_global_test_client_train_index_partition"
EXPECTED_VERSION = 2


def get_cifar_stats(data_name):
    """Return dataset class, normalization stats, and class count."""

    data_dict = {
        "cifar10": (
            CIFAR10,
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
            10,
        ),
        "cifar100": (
            CIFAR100,
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
            100,
        ),
    }
    if data_name not in data_dict:
        raise ValueError(f"Unsupported dataset: {data_name}")
    return data_dict[data_name]


def build_transforms(data_name):
    """Use augmentation for client training and plain transforms for eval."""

    _, mean, std, _ = get_cifar_stats(data_name)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, eval_transform


def _use_pin_memory(args) -> bool:
    """仅在 CUDA 训练时启用 pin_memory。"""
    return bool(getattr(args, "pin_memory", False)) and str(args.device).startswith("cuda")


def _build_loader_kwargs(args):
    num_workers = int(getattr(args, "num_workers", 0))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": _use_pin_memory(args),
    }
    if num_workers > 0:
        # 仅在 worker 模式下传入这些参数，避免 num_workers=0 时触发 PyTorch 报错。
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 2))
    return loader_kwargs


def load_partition_meta(args):
    meta_path = os.path.join(args.data_save_path, args.partition_meta_name)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing partition metadata: {meta_path}. "
            "Run `python train.py` to rebuild partition_meta.pt and partition_stats.json."
        )

    meta = torch.load(meta_path, weights_only=False)
    validate_partition_meta(meta, args)
    return meta


def validate_partition_meta(meta, args):
    """Fail fast when saved partition metadata does not match current args."""

    validate_partition_structure(meta, args)

    checks = [
        ("protocol", meta.get("protocol"), EXPECTED_PROTOCOL, "str"),
        ("version", meta.get("version"), EXPECTED_VERSION, "int"),
        ("dataset", meta.get("dataset"), args.data_name, "str"),
        ("num_clients", meta.get("num_clients"), args.num_clients, "int"),
        ("alpha", meta.get("alpha"), args.alpha, "float"),
        ("seed", meta.get("seed"), args.seed, "int"),
        ("min_datasize", meta.get("min_datasize"), args.min_datasize, "int"),
        ("data_path", meta.get("data_path"), args.data_path, "path"),
    ]

    for field, actual, expected, value_type in checks:
        if not metadata_value_matches(actual, expected, value_type):
            raise_partition_mismatch(field, actual, expected)


def validate_partition_structure(meta, args):
    """Check only top-level split structure so bad metadata fails early."""

    splits = meta.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(
            "partition_meta is incomplete: missing a valid `splits` dictionary. "
            "Run `python train.py` to rebuild partition_meta.pt and partition_stats.json."
        )

    required_split_keys = {
        "client_train_pool_indices",
        "client_train_indices",
        "global_test_indices",
    }
    missing = required_split_keys - set(splits.keys())
    if missing:
        raise ValueError(
            f"partition_meta is incomplete: missing split keys {sorted(missing)}. "
            "Run `python train.py` to rebuild partition_meta.pt and partition_stats.json."
        )

    if not isinstance(splits["client_train_indices"], dict):
        raise ValueError(
            "`splits['client_train_indices']` must be a dict. "
            "Run `python train.py` to rebuild partition_meta.pt and partition_stats.json."
        )

    expected_client_keys = {str(i) for i in range(1, args.num_clients + 1)}
    actual_client_keys = set(splits["client_train_indices"].keys())
    if actual_client_keys != expected_client_keys:
        raise ValueError(
            "partition_meta has incomplete client_train_indices keys: "
            f"expected {sorted(expected_client_keys)}, found {sorted(actual_client_keys)}. "
            "Run `python train.py` to rebuild partition_meta.pt and partition_stats.json."
        )

    for client_id, indices in splits["client_train_indices"].items():
        if not isinstance(indices, (list, tuple)):
            raise ValueError(
                f"`splits['client_train_indices']['{client_id}']` must be a list or tuple. "
                "Run `python train.py` to rebuild partition_meta.pt and partition_stats.json."
            )

    for key in ["client_train_pool_indices", "global_test_indices"]:
        if not isinstance(splits[key], (list, tuple)):
            raise ValueError(
                f"`splits['{key}']` must be a list or tuple. "
                "Run `python train.py` to rebuild partition_meta.pt and partition_stats.json."
            )


def metadata_value_matches(actual, expected, value_type):
    if actual is None:
        return False
    if value_type == "float":
        return abs(float(actual) - float(expected)) <= 1e-12
    if value_type == "int":
        return int(actual) == int(expected)
    if value_type == "path":
        return os.path.abspath(os.path.normpath(str(actual))) == os.path.abspath(os.path.normpath(str(expected)))
    return actual == expected


def raise_partition_mismatch(field, actual, expected):
    raise ValueError(
        f"partition_meta mismatch for `{field}`: found {actual!r}, expected {expected!r}. "
        "Run `python train.py` to rebuild partition_meta.pt and partition_stats.json."
    )


def build_raw_cifar_dataset(args, train, transform):
    dataset_cls, _, _, _ = get_cifar_stats(args.data_name)
    try:
        return dataset_cls(
            root=args.data_path,
            train=train,
            download=False,
            transform=transform,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not load raw CIFAR data from `{args.data_path}` with download=False. "
            "This project uses index-based partition metadata, so training still requires "
            "the original CIFAR files. Run `python train.py` to rebuild partition files "
            "after ensuring the dataset is available."
        ) from e
    except RuntimeError as e:
        error_text = str(e).lower()
        missing_keywords = ["not found", "dataset not found", "no such file", "download"]
        corrupt_keywords = ["corrupt", "corrupted", "truncate", "truncated", "invalid", "pickle", "unpickling"]

        if any(keyword in error_text for keyword in missing_keywords):
            raise FileNotFoundError(
                f"Could not load raw CIFAR data from `{args.data_path}` with download=False. "
                "This project uses index-based partition metadata, so training still requires "
                "the original CIFAR files. Run `python train.py` to rebuild partition files "
                "after ensuring the dataset is available."
            ) from e

        if any(keyword in error_text for keyword in corrupt_keywords):
            raise RuntimeError(
                f"Raw CIFAR files were found under `{args.data_path}`, but loading failed and "
                "the dataset may be corrupted or incomplete. This project uses index-based "
                "partition metadata, so training still requires the original CIFAR files. "
                "Please check the dataset files, then run `python train.py` to rebuild "
                "partition files."
            ) from e

        raise RuntimeError(
            f"Failed to load raw CIFAR data from `{args.data_path}` with download=False. "
            "This project uses index-based partition metadata, so training still requires "
            "the original CIFAR files. Please check data_path, then run `python train.py` "
            "to rebuild partition files after ensuring the dataset can be read."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error while loading raw CIFAR data from `{args.data_path}` with download=False. "
            "This project uses index-based partition metadata, so training still requires "
            "the original CIFAR files. Please check data_path, then run `python train.py`."
        ) from e


def build_index_dataset(args, split, client_id=None, meta=None):
    """Build a Dataset from raw CIFAR data plus saved partition indices."""

    meta = meta or load_partition_meta(args)
    train_transform, eval_transform = build_transforms(args.data_name)
    splits = meta["splits"]

    if split == "client_train":
        if client_id is None:
            raise ValueError("client_id is required for client_train split")
        indices = splits["client_train_indices"][str(client_id)]
        dataset = build_raw_cifar_dataset(args, train=True, transform=train_transform)
    elif split == "global_test":
        indices = splits["global_test_indices"]
        dataset = build_raw_cifar_dataset(args, train=False, transform=eval_transform)
    else:
        raise ValueError(f"Unknown split: {split}")

    return Subset(dataset, indices)


def build_client_train_loader(args, client_id, meta=None):
    dataset = build_index_dataset(
        args=args,
        split="client_train",
        client_id=client_id,
        meta=meta,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **_build_loader_kwargs(args),
    )


def build_global_eval_loader(args, split, meta=None):
    if split != "global_test":
        raise ValueError("split must be global_test")

    dataset = build_index_dataset(args=args, split=split, meta=meta)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **_build_loader_kwargs(args),
    )


def get_client_train_size(args, client_id, meta=None):
    meta = meta or load_partition_meta(args)
    return len(meta["splits"]["client_train_indices"][str(client_id)])
