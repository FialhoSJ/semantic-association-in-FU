"""Validated, versioned experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "synthetic"
    root: str = "data"
    download: bool = False
    synthetic_train_size: int = 240
    synthetic_test_size: int = 80
    synthetic_num_classes: int = 3


@dataclass(frozen=True)
class FederatedConfig:
    algorithm: str = "fedavg"
    num_clients: int = 4
    clients_per_round: int = 2
    global_rounds: int = 2
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    non_iid_alpha: float = 0.5
    min_client_samples: int = 1


@dataclass(frozen=True)
class ModelConfig:
    name: str = "simple_cnn"
    width: int = 16


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = 1
    name: str = "phase1_smoke"
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"
    results_root: str = "results"
    save_each_round: bool = False
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self) -> None:
        f, d, m = self.federated, self.dataset, self.model
        if self.schema_version != 1:
            raise ValueError("Unsupported config schema_version")
        if not self.name or Path(self.name).name != self.name:
            raise ValueError("name must be a single directory name")
        if d.name not in {"synthetic", "cifar10"}:
            raise ValueError("dataset.name must be synthetic or cifar10")
        if f.algorithm != "fedavg":
            raise ValueError("Phase 1 supports only fedavg")
        if m.name != "simple_cnn" or m.width <= 0:
            raise ValueError("Phase 1 supports simple_cnn with positive width")
        if not 1 <= f.clients_per_round <= f.num_clients:
            raise ValueError("clients_per_round must be between 1 and num_clients")
        positive_ints = (f.global_rounds, f.local_epochs, f.batch_size,
                         f.min_client_samples, self.seed + 1)
        if any(v <= 0 for v in positive_ints):
            raise ValueError("rounds, epochs, batch size, minimum samples and seed must be valid")
        if f.learning_rate <= 0 or f.non_iid_alpha <= 0:
            raise ValueError("learning_rate and non_iid_alpha must be positive")
        if d.synthetic_train_size <= 0 or d.synthetic_test_size <= 0 or d.synthetic_num_classes < 2:
            raise ValueError("synthetic dataset sizes must be positive and classes >= 2")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")


def _section(data: dict[str, Any], key: str, cls: type) -> Any:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    unknown = set(value) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown {key} keys: {sorted(unknown)}")
    return cls(**value)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML and reject unknown fields instead of silently ignoring them."""
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown = set(raw) - set(ExperimentConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    base = {k: v for k, v in raw.items() if k not in {"dataset", "federated", "model"}}
    config = ExperimentConfig(
        **base,
        dataset=_section(raw, "dataset", DatasetConfig),
        federated=_section(raw, "federated", FederatedConfig),
        model=_section(raw, "model", ModelConfig),
    )
    config.validate()
    return config
