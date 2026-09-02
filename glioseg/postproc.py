"""Connected-component post-processing. The Tier-5 ablation, inference-only.

Under lesion-wise scoring each spurious component counts as a full false-positive
lesion, so removing small "dust" can move scores materially. The official BraTS
metric code removes components of <= 50 voxels.

The catch, and the reason this is an ablation rather than a default: it also
deletes TRUE small lesions. Given the sub-2cm3 detection floor, that trade-off
is exactly what we want to measure, not assume.
"""

from __future__ import annotations

import numpy as np


def _label_components(mask: np.ndarray):
    try:
        from scipy import ndimage
        return ndimage.label(mask)
    except ImportError:
        from skimage.measure import label
        lab = label(mask, connectivity=3)
        return lab, int(lab.max())


def clean_channel(mask: np.ndarray, min_voxels: int = 50,
                  keep_largest: bool = False) -> tuple[np.ndarray, dict]:
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask, {"n_before": 0, "n_after": 0, "removed_voxels": 0}

    lab, n = _label_components(mask)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0

    if keep_largest:
        keep = {int(sizes.argmax())}
    else:
        keep = {i for i in range(1, n + 1) if sizes[i] >= min_voxels}

    out = np.isin(lab, list(keep)) if keep else np.zeros_like(mask)
    return out, {"n_before": int(n), "n_after": len(keep),
                 "removed_voxels": int(mask.sum() - out.sum())}


def clean_regions(region_bin: np.ndarray, min_voxels: int = 50,
                  keep_largest: bool = False, enforce_nesting: bool = True):
    """Apply per channel, then optionally restore ET subset TC subset WT.

    Cleaning channels independently can break the nesting invariant, which
    would make the regions mutually inconsistent -- so repair it by default.
    """
    out = np.zeros_like(region_bin, dtype=np.uint8)
    stats = {}
    names = ["WT", "TC", "ET"] + (["RC"] if region_bin.shape[0] > 3 else [])
    for i, nm in enumerate(names):
        c, s = clean_channel(region_bin[i], min_voxels, keep_largest)
        out[i] = c
        stats[nm] = s

    if enforce_nesting and out.shape[0] >= 3:
        out[1] = np.logical_and(out[1], out[0])      # TC inside WT
        out[2] = np.logical_and(out[2], out[1])      # ET inside TC
    return out, stats
