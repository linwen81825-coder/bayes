import json
import os
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch

from data.loader import get_cifar_stats


class CIFARPartitionBuilder:
    """
    Build index-based FL partitions for CIFAR10/CIFAR100.

    新协议：
        1. client_train 来自 official train set。
        2. server_meta_validation 从 official test set 中划分。
        3. global_test 是 official test set 扣除 server_meta_validation 后剩余部分。
        4. 客户端永远不能看到 official test set。
        5. 只保存 indices 和 metadata，transform 在 loader.py 中动态应用。
    """

    def __init__(self, args: SimpleNamespace):
        self.args = args
        self.data_save_path = self.args.data_save_path

        self.num_clients = self.args.num_clients
        self.data_name = self.args.data_name
        self.data_path = self.args.data_path
        self.alpha = self.args.alpha

        self.use_server_meta_validation = bool(
            getattr(self.args, "use_server_meta_validation", False)
        )
        self.server_meta_validation_size = int(
            getattr(self.args, "server_meta_validation_size", 1000)
        )
        self.server_meta_validation_balanced = bool(
            getattr(self.args, "server_meta_validation_balanced", True)
        )
        self.server_meta_validation_seed_offset = int(
            getattr(self.args, "server_meta_validation_seed_offset", 9100)
        )

        # 加载 torchvision 原始训练集和测试集。
        self.train_dataset, self.test_dataset, self.num_classes = self.load_dataset()

        self.min_datasize = self.args.min_datasize
        self.seed = self.args.seed
        self.rng = np.random.default_rng(self.seed)

        self.train_targets = np.array(self.train_dataset.targets)
        self.test_targets = np.array(self.test_dataset.targets)

    def load_dataset(self):
        """根据 data_name 加载 CIFAR10 或 CIFAR100。"""
        dataset_cls, _, _, num_classes = get_cifar_stats(self.data_name)

        # download=True 表示如果 ./data 下没有数据，会自动下载。
        train_dataset = dataset_cls(
            root=self.args.data_path,
            train=True,
            download=True,
            transform=None,
        )
        test_dataset = dataset_cls(
            root=self.args.data_path,
            train=False,
            download=True,
            transform=None,
        )

        return train_dataset, test_dataset, num_classes

    def build(self):
        """Create partition_meta.pt and partition_stats.json."""
        self.validate_args()

        # 关键修改：
        # server_meta_validation 从 official test set 中划分。
        server_meta_validation_indices = self.build_server_meta_validation_indices()
        server_meta_validation_index_set = set(server_meta_validation_indices)

        # 关键修改：
        # 客户端训练池使用完整 official train set。
        # 不再从 train set 里扣除 server_meta_validation。
        client_train_pool_indices = list(range(len(self.train_dataset)))

        # official train set 按 Dirichlet non-IID 划给客户端。
        client_train_indices = self.dirichlet_client_split(client_train_pool_indices)

        # 关键修改：
        # global_test 是 official test set 扣掉 server_meta_validation 后剩余部分。
        global_test_indices = [
            index
            for index in range(len(self.test_dataset))
            if index not in server_meta_validation_index_set
        ]

        meta = {
            "protocol": "server_meta_validation_client_train_global_test_partition",
            "version": 4,
            "dataset": self.data_name,
            "data_path": self.data_path,
            "num_classes": self.num_classes,
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "seed": self.seed,
            "min_datasize": self.min_datasize,
            "use_server_meta_validation": self.use_server_meta_validation,
            "server_meta_validation_size": self.server_meta_validation_size,
            "server_meta_validation_balanced": self.server_meta_validation_balanced,
            "server_meta_validation_seed_offset": self.server_meta_validation_seed_offset,
            "index_space": {
                "server_meta_validation": "official_test",
                "client_train": "official_train",
                "client_train_pool": "official_train",
                "global_test": "official_test_minus_server_meta_validation",
            },
            "splits": {
                "server_meta_validation_indices": server_meta_validation_indices,
                "client_train_pool_indices": client_train_pool_indices,
                "client_train_indices": {
                    str(client_id): indices
                    for client_id, indices in client_train_indices.items()
                },
                "global_test_indices": global_test_indices,
            },
        }

        stats = self.build_stats(meta)
        self.log_partition_summary(meta)
        self.save(meta, stats)

        return meta, stats

    def validate_args(self):
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")

        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        if self.min_datasize <= 0:
            raise ValueError("min_datasize must be positive")

        if self.server_meta_validation_size < 0:
            raise ValueError("server_meta_validation_size must be non-negative")

        if self.use_server_meta_validation:
            if self.server_meta_validation_size <= 0:
                raise ValueError(
                    "server_meta_validation_size must be positive when "
                    "use_server_meta_validation=true"
                )

            # 关键修改：
            # server_meta_validation 现在从 official test set 划分，
            # 所以大小要小于 official test set。
            if self.server_meta_validation_size >= len(self.test_dataset):
                raise ValueError(
                    "server_meta_validation_size must be smaller than "
                    "the official test set size"
                )

    def build_server_meta_validation_indices(self):
        """
        Build server meta-validation indices.

        新协议：
            server_meta_validation 从 official test set 中划分。
            如果 balanced=true，则 CIFAR10 下 1000 张就是每类 100 张。
        """
        if not self.use_server_meta_validation:
            return []

        meta_seed = int(self.seed) + int(self.server_meta_validation_seed_offset)
        rng = np.random.default_rng(meta_seed)

        total_size = int(self.server_meta_validation_size)

        if not self.server_meta_validation_balanced:
            all_indices = np.arange(len(self.test_dataset))
            return rng.choice(
                all_indices,
                size=total_size,
                replace=False,
            ).astype(int).tolist()

        base_per_class = total_size // self.num_classes
        remainder = total_size % self.num_classes

        selected_indices = []

        for class_id in range(self.num_classes):
            take_count = base_per_class + (1 if class_id < remainder else 0)

            if take_count <= 0:
                continue

            # 关键修改：
            # 从 official test targets 里按类别抽 meta-validation。
            class_indices = np.where(self.test_targets == class_id)[0]

            if class_indices.size < take_count:
                raise ValueError(
                    f"Not enough official test samples in class {class_id} "
                    f"to build server_meta_validation set."
                )

            shuffled_class_indices = rng.permutation(class_indices)
            selected_indices.extend(
                shuffled_class_indices[:take_count].astype(int).tolist()
            )

        return selected_indices

    def dirichlet_client_split(self, pool_indices, max_attempts=100):
        """Split client_train_pool into non-IID client train indices."""
        pool_indices = np.array(pool_indices)
        pool_targets = self.train_targets[pool_indices]

        class_indices = [
            pool_indices[np.where(pool_targets == class_id)[0]]
            for class_id in range(self.num_classes)
        ]

        for _ in range(max_attempts):
            client_indices = {client_id: [] for client_id in range(1, self.num_clients + 1)}

            label_distribution = self.rng.dirichlet(
                [self.alpha] * self.num_clients,
                self.num_classes,
            )

            for class_id, class_idcs in enumerate(class_indices):
                shuffled_idcs = self.rng.permutation(class_idcs)

                split_points = (
                    np.cumsum(label_distribution[class_id])[:-1] * len(shuffled_idcs)
                ).astype(int)

                for client_id, idcs in enumerate(
                    np.split(shuffled_idcs, split_points),
                    start=1,
                ):
                    client_indices[client_id].extend(idcs.tolist())

            for idcs in client_indices.values():
                self.rng.shuffle(idcs)

            if min(len(idcs) for idcs in client_indices.values()) >= self.min_datasize:
                return client_indices

        raise ValueError(
            "Unable to split data with the requested min_datasize. "
            "Try increasing alpha, reducing num_clients, or lowering min_datasize "
            "in the YAML config files."
        )

    def build_stats(self, meta):
        splits = meta["splits"]

        client_class_counts = {
            client_id: self.class_counts(indices, self.train_targets)
            for client_id, indices in splits["client_train_indices"].items()
        }

        return {
            "protocol": meta["protocol"],
            "version": meta["version"],
            "dataset": self.data_name,
            "num_classes": self.num_classes,
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "seed": self.seed,
            "min_datasize": self.min_datasize,
            "use_server_meta_validation": self.use_server_meta_validation,
            "server_meta_validation_size": self.server_meta_validation_size,
            "server_meta_validation_balanced": self.server_meta_validation_balanced,
            "server_meta_validation_seed_offset": self.server_meta_validation_seed_offset,
            "sizes": {
                "official_train": len(self.train_dataset),
                "official_test": len(self.test_dataset),
                "server_meta_validation": len(
                    splits["server_meta_validation_indices"]
                ),
                "client_train_pool": len(splits["client_train_pool_indices"]),
                "global_test": len(splits["global_test_indices"]),
                "client_train": {
                    client_id: len(indices)
                    for client_id, indices in splits["client_train_indices"].items()
                },
            },
            "class_counts": {
                # 关键修改：
                # server_meta_validation 来自 official test set，所以使用 test_targets。
                "server_meta_validation": self.class_counts(
                    splits["server_meta_validation_indices"],
                    self.test_targets,
                ),
                "client_train_pool": self.class_counts(
                    splits["client_train_pool_indices"],
                    self.train_targets,
                ),
                "global_test": self.class_counts(
                    splits["global_test_indices"],
                    self.test_targets,
                ),
                "client_train": client_class_counts,
            },
        }

    def class_counts(self, indices, targets):
        counts = Counter(int(targets[index]) for index in indices)
        return {
            str(class_id): int(counts.get(class_id, 0))
            for class_id in range(self.num_classes)
        }

    def log_partition_summary(self, meta):
        splits = meta["splits"]

        # 关键修改：
        # server_meta_validation 来自 official test set，所以使用 test_targets。
        server_meta_counts = self.class_counts(
            splits["server_meta_validation_indices"],
            self.test_targets,
        )

        server_meta_counts_for_log = {
            int(class_id): count
            for class_id, count in server_meta_counts.items()
        }

        print(
            f"--use_server_meta_validation : "
            f"{str(self.use_server_meta_validation).lower()}"
        )
        print("--server_meta_validation_source : official_test")
        print(
            f"--server_meta_validation_size : "
            f"{len(splits['server_meta_validation_indices'])}"
        )
        print(
            f"--server_meta_validation_class_counts : "
            f"{server_meta_counts_for_log}"
        )
        print(
            f"--client_train_pool_size : "
            f"{len(splits['client_train_pool_indices'])}"
        )
        print(
            f"--global_test_size : "
            f"{len(splits['global_test_indices'])}"
        )

    def save(self, meta, stats):
        os.makedirs(self.data_save_path, exist_ok=True)

        meta_path = os.path.join(self.data_save_path, self.args.partition_meta_name)
        stats_path = os.path.join(self.data_save_path, self.args.partition_stats_name)

        torch.save(meta, meta_path)

        with open(stats_path, "w", encoding="utf-8") as stats_file:
            json.dump(stats, stats_file, ensure_ascii=False, indent=2)

        print(f"Saved partition meta to {meta_path}")
        print(f"Saved partition stats to {stats_path}")