"""Sequential FL simulation: independent local training and weighted FedAvg."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from safu.config import FederatedConfig
from safu.model import BaseModel


State = dict[str, torch.Tensor]


@dataclass(frozen=True)
class ClientUpdate:
    client_id: int
    num_samples: int
    state: State
    train_loss: float
    duration_seconds: float


def state_bytes(state: Mapping[str, torch.Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in state.values())


def fedavg(updates: list[ClientUpdate]) -> State:
    """Sample-weighted aggregation. Integer buffers use the largest client state."""
    if not updates or any(update.num_samples <= 0 for update in updates):
        raise ValueError("FedAvg requires nonempty updates with positive sample counts")
    keys = set(updates[0].state)
    if any(set(update.state) != keys for update in updates):
        raise ValueError("Client state dictionaries differ")
    total = sum(update.num_samples for update in updates)
    largest = max(updates, key=lambda update: update.num_samples)
    result: State = {}
    for name, reference in updates[0].state.items():
        if any(update.state[name].shape != reference.shape for update in updates):
            raise ValueError(f"Client tensor shapes differ for {name}")
        if not torch.is_floating_point(reference):
            result[name] = largest.state[name].clone()
            continue
        accumulator = torch.zeros_like(reference, dtype=torch.float64)
        for update in updates:
            accumulator += update.state[name].to(dtype=torch.float64) * (update.num_samples / total)
        result[name] = accumulator.to(dtype=reference.dtype)
    return result


def train_client(
    client_id: int,
    global_model: BaseModel,
    dataset: Dataset,
    indices: np.ndarray,
    config: FederatedConfig,
    device: torch.device,
    seed: int,
) -> ClientUpdate:
    started = time.perf_counter()
    model = copy.deepcopy(global_model).to(device)
    model.train()
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    examples_seen = 0
    for _ in range(config.local_epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            examples_seen += len(labels)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return ClientUpdate(client_id, len(indices), state, loss_sum / examples_seen,
                        time.perf_counter() - started)


@torch.no_grad()
def evaluate(model: BaseModel, dataset: Dataset, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    loss_sum = 0.0
    correct = 0
    total = 0
    confusion: torch.Tensor | None = None
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if confusion is None:
            confusion = torch.zeros(logits.shape[1], logits.shape[1], dtype=torch.int64)
        labels_cpu = labels.cpu()
        predictions_cpu = logits.argmax(dim=1).cpu()
        confusion += torch.bincount(
            labels_cpu * logits.shape[1] + predictions_cpu,
            minlength=logits.shape[1] ** 2,
        ).reshape(logits.shape[1], logits.shape[1])
        loss_sum += criterion(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    if total == 0:
        raise ValueError("Evaluation dataset is empty")
    assert confusion is not None
    true_positive = confusion.diag().to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    actual = confusion.sum(dim=1).to(torch.float64)
    precision = torch.where(predicted > 0, true_positive / predicted.clamp_min(1), 0)
    recall = torch.where(actual > 0, true_positive / actual.clamp_min(1), 0)
    f1 = torch.where(precision + recall > 0,
                     2 * precision * recall / (precision + recall).clamp_min(1e-12), 0)
    return {
        "test_loss": loss_sum / total,
        "test_accuracy": correct / total,
        "test_precision_macro": precision.mean().item(),
        "test_recall_macro": recall.mean().item(),
        "test_f1_macro": f1.mean().item(),
    }


class FederatedServer:
    def __init__(
        self, model: BaseModel, train_dataset: Dataset, test_dataset: Dataset,
        partitions: list[np.ndarray], config: FederatedConfig,
        device: torch.device, seed: int,
    ) -> None:
        if len(partitions) != config.num_clients:
            raise ValueError("Partition count must equal num_clients")
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.partitions = partitions
        self.config = config
        self.device = device
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run_round(self, round_number: int) -> dict[str, float | int | str]:
        started = time.perf_counter()
        selected = sorted(self.rng.choice(
            len(self.partitions), size=self.config.clients_per_round, replace=False
        ).tolist())
        before = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        download_bytes = state_bytes(before) * len(selected)
        updates = [
            train_client(
                client_id, self.model, self.train_dataset, self.partitions[client_id],
                self.config, self.device, self.seed + round_number * 100000 + client_id,
            ) for client_id in selected
        ]
        self.model.load_state_dict(fedavg(updates))
        scores = evaluate(self.model, self.test_dataset, self.config.batch_size, self.device)
        total_samples = sum(update.num_samples for update in updates)
        return {
            "round": round_number,
            "selected_clients": ",".join(map(str, selected)),
            "train_loss": sum(update.train_loss * update.num_samples for update in updates) / total_samples,
            **scores,
            "communication_bytes": download_bytes + sum(state_bytes(update.state) for update in updates),
            "local_train_seconds": sum(update.duration_seconds for update in updates),
            "round_seconds": time.perf_counter() - started,
        }
