from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from command_classifier.config import BCRESNET_TAU
from command_classifier.model.bcresnet import BCResNets


def build_model(pretrained: bool = False, num_classes: int = 1) -> nn.Module:
    """
    Build a BCResNet binary classifier.

    Input: (batch, 1, N_MELS, T) — single-channel log-mel spectrogram.
    Output: raw logit of shape (batch, 1).

    `pretrained` is ignored (BCResNet trains from scratch).
    """
    base_c = int(BCRESNET_TAU * 8)
    return BCResNets(base_c=base_c, num_classes=num_classes)


def freeze_backbone(model: nn.Module) -> None:
    """No-op: BCResNet trains from scratch with no separate freeze phase."""
    pass


def unfreeze_backbone(model: nn.Module) -> None:
    pass


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def get_param_groups(model: nn.Module, head_lr: float, backbone_lr_factor: float) -> list[dict[str, Any]]:
    """Return all BCResNet params in a single group."""
    return [{"params": _unwrap(model).parameters(), "lr": head_lr}]


def prepare_model(model: nn.Module, device_ids: Optional[list[int]] = None) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=device_ids)
    return model
