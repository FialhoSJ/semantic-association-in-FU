"""Dataset creation and reproducible label-skew Dirichlet partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from safu.config import DatasetConfig


@dataclass(frozen=True)
class DataBundle:
    train: Dataset
    test: Dataset
    train_labels: np.ndarray
    num_classes: int


def _synthetic(size: int, classes: int, seed: int) -> TensorDataset:
    """Small learnable image task for pipeline smoke tests, not research results."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(size, dtype=torch.long) % classes
    labels = labels[torch.randperm(size, generator=generator)]
    images = 0.15 * torch.randn(size, 3, 32, 32, generator=generator)
    for index, label in enumerate(labels.tolist()):
        images[index, label % 3, 4:20, 4:20] += 1.0
        images[index, (label // 3) % 3, 20:28, 20:28] += 0.4
    return TensorDataset(images, labels)


def load_data(config: DatasetConfig, seed: int) -> DataBundle:
    if config.name == "synthetic":
        train = _synthetic(config.synthetic_train_size, config.synthetic_num_classes, seed)
        test = _synthetic(config.synthetic_test_size, config.synthetic_num_classes, seed + 1)
        return DataBundle(train, test, train.tensors[1].numpy(), config.synthetic_num_classes)
    if config.name == "cifar10":
        from torchvision import datasets, transforms

        transform = transforms.ToTensor()
        train = datasets.CIFAR10(config.root, train=True, transform=transform, download=config.download)
        test = datasets.CIFAR10(config.root, train=False, transform=transform, download=config.download)
        return DataBundle(train, test, np.asarray(train.targets, dtype=np.int64), 10)
    raise ValueError(f"Unsupported dataset: {config.name}")


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples: int = 1,
    max_attempts: int = 1000,
) -> list[np.ndarray]:
    """Assign each training index exactly once; retry if a client is too small."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) < num_clients * min_samples:
        raise ValueError("Not enough one-dimensional labels for the client minimum")
    if num_clients < 1 or alpha <= 0 or min_samples < 1:
        raise ValueError("Invalid Dirichlet partition parameters")
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(max_attempts):
        parts: list[list[int]] = [[] for _ in range(num_clients)]
        for label in classes:
            indices = rng.permutation(np.flatnonzero(labels == label))
            weights = rng.dirichlet(np.full(num_clients, alpha))
            counts = rng.multinomial(len(indices), weights)
            offset = 0
            for client_id, count in enumerate(counts):
                parts[client_id].extend(indices[offset:offset + count].tolist())
                offset += count
        if all(len(part) >= min_samples for part in parts):
            return [rng.permutation(np.asarray(part, dtype=np.int64)) for part in parts]
    raise ValueError("Could not satisfy min_client_samples; increase alpha or reduce minimum")
