"""Small baseline model with a stable representation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from safu.config import ModelConfig


class BaseModel(nn.Module, ABC):
    """A model can expose embeddings without tying future semantics to its layers."""

    @abstractmethod
    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return one flat representation per image."""

    @abstractmethod
    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return class logits from representations."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(images))


class SimpleCNN(BaseModel):
    def __init__(self, num_classes: int, width: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.head = nn.Linear(width * 2 * 4 * 4, num_classes)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


def build_model(config: ModelConfig, num_classes: int) -> BaseModel:
    if config.name == "simple_cnn":
        return SimpleCNN(num_classes, config.width)
    raise ValueError(f"Unsupported model: {config.name}")
