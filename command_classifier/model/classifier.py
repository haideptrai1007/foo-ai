from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn

try:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
except ImportError:  # pragma: no cover
    MobileNet_V3_Small_Weights = None  # type: ignore[assignment]
    mobilenet_v3_small = None  # type: ignore[assignment]

from command_classifier.config import DROPOUT


def _require_torchvision() -> None:
    if mobilenet_v3_small is None or MobileNet_V3_Small_Weights is None:  # type: ignore[truthy-function]
        raise ImportError("torchvision is required for model building. Install torchvision to train/infer.")


def build_model(pretrained: bool = True, num_classes: int = 1, dropout: float = DROPOUT) -> nn.Module:
    """
    Build a MobileNetV3-Small binary classifier.

    The model keeps the original `features` backbone (expects 3-channel input,
    which we prepare in preprocessing by repeating the spectrogram channel).
    """

    _require_torchvision()
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)

    # Replace classifier head. Preserve backbone features untouched.
    if not hasattr(model, "classifier"):
        raise RuntimeError("Unexpected MobileNetV3 model: missing `classifier` attribute.")

    # torchvision's classifier starts with Linear(in_features -> 1024)
    first = model.classifier[0]
    if not isinstance(first, nn.Linear):
        in_features = getattr(first, "in_features", None)
    else:
        in_features = first.in_features
    if in_features is None:
        raise RuntimeError("Could not infer classifier in_features.")

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=float(dropout)),
        nn.Linear(256, int(num_classes)),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze everything except the classifier head.

    Works with both plain models and DataParallel-wrapped models.
    """

    module = model.module if isinstance(model, nn.DataParallel) else model
    for param in getattr(module, "features").parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    module = model.module if isinstance(model, nn.DataParallel) else model
    for param in getattr(module, "features").parameters():
        param.requires_grad = True


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def get_param_groups(model: nn.Module, head_lr: float, backbone_lr_factor: float) -> list[dict[str, Any]]:
    """
    Return two param groups for optimizer.
    """

    module = _unwrap(model)
    return [
        {"params": module.features.parameters(), "lr": head_lr * backbone_lr_factor},
        {"params": module.classifier.parameters(), "lr": head_lr},
    ]


def prepare_model(model: nn.Module, device_ids: Optional[list[int]] = None) -> nn.Module:
    """
    Optionally wrap with DataParallel and move to the right device.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=device_ids)
    return model

