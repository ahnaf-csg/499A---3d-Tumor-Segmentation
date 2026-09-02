"""Sliding-window inference and per-case scoring.

Metrics are computed PER CHANNEL from sigmoid probabilities. There is no argmax
anywhere in this file -- the regions are nested and overlapping, so an argmax
would silently collapse them.
"""

from __future__ import annotations

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from .config import Config
from .metrics import aggregate, score_case, voxel_volume_mm3
from .postproc import clean_regions


def _affine_of(batch) -> np.ndarray:
    """Voxel geometry from the loaded image metadata, never hardcoded."""
    for key in ("label", "image"):
        t = batch.get(key)
        meta = getattr(t, "meta", None)
        if meta is not None and "affine" in meta:
            a = meta["affine"]
            a = a[0] if getattr(a, "ndim", 2) == 3 else a
            return np.asarray(a.cpu() if hasattr(a, "cpu") else a, dtype=float)
    md = batch.get("label_meta_dict") or batch.get("image_meta_dict")
    if md and "affine" in md:
        a = md["affine"]
        a = a[0] if getattr(a, "ndim", 2) == 3 else a
        return np.asarray(a.cpu() if hasattr(a, "cpu") else a, dtype=float)
    return np.eye(4)


@torch.no_grad()
def evaluate(model, loader, cfg: Config, with_ap: bool = True,
             device: str | None = None, thresh: float = 0.5):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    records = []

    for batch in loader:
        x = batch["image"].to(dev)
        gt = batch["label"]

        with torch.autocast(dev, dtype=torch.float16,
                            enabled=cfg.amp and dev == "cuda"):
            logits = sliding_window_inference(
                x, cfg.patch_size, cfg.sw_batch_size, model,
                overlap=cfg.sw_overlap, mode="gaussian")
        probs = torch.sigmoid(logits.float())[0].cpu().numpy()
        gt_np = gt[0].cpu().numpy().astype(np.uint8)

        if cfg.postproc_min_voxels > 0:
            binm, _ = clean_regions(probs >= thresh, cfg.postproc_min_voxels)
            probs_for_score = binm.astype(np.float32)   # already thresholded
            rec = score_case(probs_for_score, gt_np, thresh=0.5, with_ap=False)
            if with_ap:                                  # AP needs soft probs
                soft = score_case(probs, gt_np, thresh=thresh, with_ap=True)
                for k in soft:
                    if k.endswith("_AP") or k == "mAP":
                        rec[k] = soft[k]
        else:
            rec = score_case(probs, gt_np, thresh=thresh, with_ap=with_ap)

        mm3 = voxel_volume_mm3(_affine_of(batch))
        rec["voxel_mm3"] = mm3
        rec["gt_volume_cm3"] = float(gt_np[2].sum() * mm3 / 1000.0)   # ET channel
        c = batch.get("case", ["?"])
        rec["case"] = c[0] if isinstance(c, (list, tuple)) else str(c)
        records.append(rec)

    return aggregate(records), records


@torch.no_grad()
def predict_volume(model, batch, cfg: Config, device=None, thresh=0.5):
    """Single-case inference -> (probs, binary, gt). For XAI and figures."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x = batch["image"].to(dev)
    with torch.autocast(dev, dtype=torch.float16, enabled=cfg.amp and dev == "cuda"):
        logits = sliding_window_inference(x, cfg.patch_size, cfg.sw_batch_size,
                                          model, overlap=cfg.sw_overlap, mode="gaussian")
    probs = torch.sigmoid(logits.float())[0].cpu().numpy()
    return probs, (probs >= thresh).astype(np.uint8), batch["label"][0].cpu().numpy()
