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


class TverskyBCELoss(nn.Module):
    """Per-channel Tversky + BCE-with-logits.

    Tversky index generalises Dice by weighting the two error types separately:

        TI = TP / (TP + alpha*FP + beta*FN)

    alpha = beta = 0.5 recovers Dice exactly.

    DIRECTION MATTERS AND IS COUNTERINTUITIVE. Salehi et al. (arXiv:1706.05721)
    tuned beta > alpha because their MS-lesion models UNDER-segmented and they
    wanted recall. Our models OVER-segment (recall 0.92 vs precision 0.74), so we
    need the opposite: alpha > beta, penalising false positives harder to buy
    precision back.

    Reference: Salehi, Erdogmus & Gholipour, "Tversky loss function for image
    segmentation using 3D fully convolutional deep networks", MLMI/MICCAI 2017,
    arXiv:1706.05721. Focal variant: Abraham & Khan, ISBI 2019, arXiv:1810.07842.
    """

    def __init__(self, alpha: float = 0.7, beta: float | None = None,
                 gamma: float | None = None, bce_weight: float = 1.0,
                 smooth_nr: float = 1e-5, smooth_dr: float = 1e-5):
        super().__init__()
        if beta is None:
            beta = 1.0 - alpha          # convention: alpha + beta = 1
        self.alpha, self.beta = alpha, beta
        self.gamma = gamma              # None = plain Tversky; set ~4/3 for focal
        self.bce_weight = bce_weight
        self.smooth_nr, self.smooth_dr = smooth_nr, smooth_dr

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.float()
        probs = torch.sigmoid(logits)
        dims = tuple(range(2, probs.ndim))          # spatial only -> (B, C)

        tp = (probs * target).sum(dims)
        fp = (probs * (1 - target)).sum(dims)
        fn = ((1 - probs) * target).sum(dims)

        ti = (tp + self.smooth_nr) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth_dr)
        loss_t = 1.0 - ti
        if self.gamma is not None:                  # Focal Tversky
            loss_t = loss_t.clamp(min=1e-6) ** self.gamma

        bce = F.binary_cross_entropy_with_logits(logits, target)
        return loss_t.mean() + self.bce_weight * bce


class DiceOnly(nn.Module):
    """Ablation control: Dice with no cross-entropy term."""

    def __init__(self, **kw):
        super().__init__()
        self.d = DiceBCELoss(alpha=1.0, beta=0.0, **kw)

    def forward(self, logits, target):
        return self.d(logits, target)


class BlobDiceBCELoss(nn.Module):
    """Instance-imbalance-aware loss: every lesion counts equally.

    Reference: Kofler et al., "blob loss: instance imbalance aware loss
    functions for semantic segmentation", IPMI 2023, arXiv:2205.08209.
    Reported ~5% F1 improvement on MS lesions and ~3% on liver tumour.

    THE PROBLEM IT TARGETS. Global Dice is a single overlap ratio over the whole
    volume, so a 40 cm3 lesion contributes ~400x more to the numerator than a
    0.1 cm3 one. Gradient is therefore dominated by large lesions and small ones
    are effectively ignored -- which is exactly the failure measured here
    (ET Dice 0.84 above 15 cm3 versus 0.05 below 0.5 cm3).

    THE MECHANISM. Label the ground-truth connected components. For each
    component in turn, compute Dice against a target containing ONLY that
    component, with all OTHER components masked out of both prediction and
    target so they neither reward nor penalise. Average over components, so a
    tiny lesion carries the same weight as a huge one. Add the global term back
    to retain overall calibration:

        L = alpha * Dice_global + beta * mean_i(Dice_i) + gamma * BCE

    COST. Connected-component labelling runs on CPU per batch item per channel.
    At 64^3 that is a few milliseconds, so the overhead is small but not free.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 1.0,
                 smooth: float = 1e-5, max_blobs: int = 24,
                 channels: tuple[int, ...] | None = None):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.smooth = smooth
        self.max_blobs = max_blobs        # cap cost on pathological cases
        # which channels get the blob term. ET (index 2) is the small-structure
        # one; applying it to WT adds cost for little benefit.
        self.channels = channels

    @staticmethod
    def _label(mask_np):
        try:
            from scipy import ndimage
            return ndimage.label(mask_np)
        except ImportError:
            from skimage.measure import label
            lab = label(mask_np, connectivity=3)
            return lab, int(lab.max())

    def _dice(self, p, t):
        inter = (p * t).sum()
        return (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.float()
        probs = torch.sigmoid(logits)
        B, C = probs.shape[:2]
        dims = tuple(range(2, probs.ndim))

        # global region term
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        global_dice = (1.0 - (2 * inter + self.smooth) / (denom + self.smooth)).mean()

        chans = self.channels if self.channels is not None else range(C)
        blob_terms = []
        for b in range(B):
            for c in chans:
                t_np = target[b, c].detach().cpu().numpy() > 0.5
                if not t_np.any():
                    continue
                lab, n = self._label(t_np)
                if n == 0:
                    continue
                lab_t = torch.as_tensor(lab, device=probs.device)
                # largest blobs first, so the cap keeps the meaningful ones
                sizes = torch.bincount(lab_t.flatten())[1:]
                order = torch.argsort(sizes, descending=True)[:self.max_blobs] + 1
                for bid in order.tolist():
                    this = (lab_t == bid)
                    others = (lab_t > 0) & (~this)
                    keep = ~others                  # mask other lesions out entirely
                    blob_terms.append(
                        1.0 - self._dice(probs[b, c][keep], this[keep].float()))

        blob_loss = (torch.stack(blob_terms).mean() if blob_terms
                     else torch.zeros((), device=probs.device))
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return self.alpha * global_dice + self.beta * blob_loss + self.gamma * bce


def build_loss(name: str, **kw) -> nn.Module:
    """Loss registry. Names ending in a number encode alpha, e.g. 'tversky70'
    means alpha=0.70 (favouring precision), so the ablation table reads cleanly.
    """
    if name == "dice_bce":
        return DiceBCELoss(**kw)
    if name == "dice_focal":
        return DiceFocalLoss(**kw)
    if name == "dice":
        return DiceOnly(**kw)
    if name.startswith("blob"):
        # 'blob' applies the instance term to ET only (channel 2); 'blob_all'
        # applies it to every region. ET-only is the cheaper default and targets
        # the structure that actually fails.
        return BlobDiceBCELoss(channels=None if name.endswith("_all") else (2,), **kw)
    if name.startswith("tversky"):
        suffix = name[len("tversky"):].rstrip("f")
        alpha = float(suffix) / 100.0 if suffix else 0.7
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), parsed {alpha} from '{name}'")
        focal = name.endswith("f")
        return TverskyBCELoss(alpha=alpha, gamma=(4/3 if focal else None), **kw)
    raise ValueError(f"unknown loss '{name}'")
