import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

from configs.args import parse
from data.loader import get_cifar_stats
from utils.utils import set_seed


class CIFARPartitionBuilder:
    """Build index-based FL partitions for CIFAR10/CIFAR100.

    Protocol:
    1. Official train set is split stratified into global_val and federated_train_pool.
    2. federated_train_pool is split across clients with Dirichlet non-IID sampling.
    3. Official test set is kept as global_test and is never used by clients.
    4. Only indices and metadata are saved; transforms are applied dynamically in loader.py.
    """

    def __init__(self,args:argparse.Namespace):
        self.args = args
        self.data_save_path = self.args.data_save_path
        self.num_clients = self.args.num_clients
        self.data_name = self.args.data_name
        self.data_path = self.args.data_path
        self.alpha = self.args.alpha

        # 先加载 torchvision 里的原始训练集和测试集。
        self.train_dataset,self.test_dataset,self.num_classes = self.load_dataset()

        # 官方 train set 只划给客户端；官方 test set 保持统一，不参与客户端划分。
        self.min_datasize = self.args.min_datasize
        self.global_val_ratio = self.args.global_val_ratio
        self.seed = self.args.seed
        self.rng = np.random.default_rng(self.seed)
        self.train_targets = np.array(self.train_dataset.targets)
        self.test_targets = np.array(self.test_dataset.targets)

    def load_dataset(self):
        """根据 data_name 加载 CIFAR10 或 CIFAR100。"""

        dataset_cls, _, _, num_classes = get_cifar_stats(self.data_name)

        # download=True 表示如果 ./data 下没有数据，会自动下载。
        train_dataset = dataset_cls(root=self.args.data_path, train=True, download=True, transform=None)
        test_dataset = dataset_cls(root=self.args.data_path, train=False, download=True, transform=None)
        return train_dataset, test_dataset, num_classes

    def build(self):
        """Create partition_meta.pt and partition_stats.json."""

        self.validate_args()
        global_val_indices, federated_train_pool_indices = self.stratified_global_val_split()
        client_train_indices = self.dirichlet_client_split(federated_train_pool_indices)

        meta = {
            "protocol": "server_global_val_client_train_index_partition",
            "version": 1,
            "dataset": self.data_name,
            "data_path": self.data_path,
            "num_classes": self.num_classes,
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "seed": self.seed,
            "global_val_ratio": self.global_val_ratio,
            "min_datasize": self.min_datasize,
            "index_space": {
                "client_train": "official_train",
                "global_val": "official_train",
                "global_test": "official_test",
            },
            "splits": {
                "global_val_indices": global_val_indices,
                "federated_train_pool_indices": federated_train_pool_indices,
                "client_train_indices": {
                    str(client_id): indices
                    for client_id, indices in client_train_indices.items()
                },
                "global_test_indices": list(range(len(self.test_dataset))),
            },
        }
        stats = self.build_stats(meta)
        self.save(meta, stats)
        return meta, stats

    def validate_args(self):
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not (0 < self.global_val_ratio < 1):
            raise ValueError("global_val_ratio must be between 0 and 1")
        if self.min_datasize <= 0:
            raise ValueError("min_datasize must be positive")

    def stratified_global_val_split(self):
        """Split official train indices into global_val and federated_train_pool."""

        global_val_indices = []
        federated_train_pool_indices = []

        for class_id in range(self.num_classes):
            class_indices = np.where(self.train_targets == class_id)[0]
            class_indices = self.rng.permutation(class_indices)
            val_size = int(round(len(class_indices) * self.global_val_ratio))
            val_size = min(max(val_size, 1), len(class_indices) - 1)

            global_val_indices.extend(class_indices[:val_size].tolist())
            federated_train_pool_indices.extend(class_indices[val_size:].tolist())

        self.rng.shuffle(global_val_indices)
        self.rng.shuffle(federated_train_pool_indices)
        return global_val_indices, federated_train_pool_indices

    def dirichlet_client_split(self, pool_indices, max_attempts=100):
        """Split federated_train_pool into non-IID client train indices."""

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
                for client_id, idcs in enumerate(np.split(shuffled_idcs, split_points), start=1):
                    client_indices[client_id].extend(idcs.tolist())

            for idcs in client_indices.values():
                self.rng.shuffle(idcs)

            if min(len(idcs) for idcs in client_indices.values()) >= self.min_datasize:
                return client_indices

        raise ValueError(
            "Unable to split data with the requested min_datasize. "
            "Try increasing --alpha, reducing --num_clients, or lowering --min_datasize."
        )

    def build_stats(self, meta):
        splits = meta["splits"]
        client_class_counts = {
            client_id: self.class_counts(indices, self.train_targets)
            for client_id, indices in splits["client_train_indices"].items()
        }

        return {
            "protocol": meta["protocol"],
            "dataset": self.data_name,
            "num_classes": self.num_classes,
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "seed": self.seed,
            "global_val_ratio": self.global_val_ratio,
            "sizes": {
                "official_train": len(self.train_dataset),
                "federated_train_pool": len(splits["federated_train_pool_indices"]),
                "global_val": len(splits["global_val_indices"]),
                "global_test": len(splits["global_test_indices"]),
                "client_train": {
                    client_id: len(indices)
                    for client_id, indices in splits["client_train_indices"].items()
                },
            },
            "class_counts": {
                "federated_train_pool": self.class_counts(
                    splits["federated_train_pool_indices"],
                    self.train_targets,
                ),
                "global_val": self.class_counts(splits["global_val_indices"], self.train_targets),
                "global_test": self.class_counts(splits["global_test_indices"], self.test_targets),
                "client_train": client_class_counts,
            },
        }

    def class_counts(self, indices, targets):
        counts = Counter(int(targets[index]) for index in indices)
        return {str(class_id): int(counts.get(class_id, 0)) for class_id in range(self.num_classes)}

    def save(self, meta, stats):
        os.makedirs(self.data_save_path, exist_ok=True)
        meta_path = os.path.join(self.data_save_path, self.args.partition_meta_name)
        stats_path = os.path.join(self.data_save_path, self.args.partition_stats_name)

        torch.save(meta, meta_path)
        with open(stats_path, "w", encoding="utf-8") as stats_file:
            json.dump(stats, stats_file, ensure_ascii=False, indent=2)

        print(f"Saved partition meta to {meta_path}")
        print(f"Saved partition stats to {stats_path}")

if __name__ == '__main__':
    # 直接运行 `python -m data.data` 时，会根据 --data_name 选择对应的数据处理类。
    args = parse.parse_args()
    set_seed(args.seed)
    CIFARPartitionBuilder(args=args).build()
