"""Losses for nested-region segmentation.

Two things that are easy to get wrong and silently degrade results:

1. PER-CHANNEL Dice, not flattened. Flattening the prediction and target across
   all channels collapses WT/TC/ET into one scalar overlap, so the nesting
   structure buys nothing and small regions (ET) are drowned by large ones (WT).

2. BCE **with logits**, never sigmoid-then-BCE. Under fp16 autocast the latter
   is numerically unstable; the fused version is autocast-safe.

Dice also needs a smooth term in BOTH numerator and denominator, because an
all-background patch gives 0/0 otherwise -- common with RandCropByPosNegLabel
at neg ratio > 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """alpha * per-channel soft Dice + beta * BCEWithLogits.

    logits : (B, C, ...) raw, no activation
    target : (B, C, ...) binary, same shape -- nested channels from regions.py
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                 smooth_nr: float = 1e-5, smooth_dr: float = 1e-5,
                 channel_weights: tuple[float, ...] | None = None):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.smooth_nr, self.smooth_dr = smooth_nr, smooth_dr
        self.channel_weights = channel_weights

    def dice_per_channel(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # reduce over spatial dims only -> (B, C)
        dims = tuple(range(2, probs.ndim))
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth_nr) / (denom + self.smooth_dr)
        return 1.0 - dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # fp32 for the loss even under autocast -- Dice/BCE NaN in fp16
        logits = logits.float()
        target = target.float()
        probs = torch.sigmoid(logits)

        d = self.dice_per_channel(probs, target)            # (B, C)
        if self.channel_weights is not None:
            w = torch.tensor(self.channel_weights, device=d.device, dtype=d.dtype)
            d = d * w / w.sum() * len(w)
        dice = d.mean()

        bce = F.binary_cross_entropy_with_logits(logits, target)
        return self.alpha * dice + self.beta * bce


class DiceFocalLoss(nn.Module):
    """Per-channel Dice + focal. The Tier-5 ablation arm.

    Focal (Lin et al., ICCV 2017, arXiv:1708.02002) down-weights easy voxels,
    which matters here because background dominates a 64^3 patch.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 2.0,
                 smooth_nr: float = 1e-5, smooth_dr: float = 1e-5):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.dice = DiceBCELoss(alpha=1.0, beta=0.0,
                                smooth_nr=smooth_nr, smooth_dr=smooth_dr)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.float()
        dice = self.dice(logits, target)

        # focal on top of BCE-with-logits, computed stably
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)      # prob of the true class
        focal = ((1 - p_t).clamp(min=1e-6) ** self.gamma * bce).mean()

        return self.alpha * dice + self.beta * focal


class DiceOnly(nn.Module):
    """Ablation control: Dice with no cross-entropy term."""

    def __init__(self, **kw):
        super().__init__()
        self.d = DiceBCELoss(alpha=1.0, beta=0.0, **kw)

    def forward(self, logits, target):
        return self.d(logits, target)


def build_loss(name: str, **kw) -> nn.Module:
    if name == "dice_bce":
        return DiceBCELoss(**kw)
    if name == "dice_focal":
        return DiceFocalLoss(**kw)
    if name == "dice":
        return DiceOnly(**kw)
    raise ValueError(f"unknown loss '{name}'")
