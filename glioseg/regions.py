"""Canonical label remapping and nested-region conversion.

Why sigmoid and not softmax
---------------------------
The BraTS evaluation regions are NESTED: ET subset TC subset WT. A single voxel
legitimately belongs to all three. A softmax head forces exactly one label per
voxel, which contradicts the endpoint definition and makes the outputs
incomparable to every published ET/TC/WT number.

So: 3 (or 4) INDEPENDENT sigmoid channels, BCE-with-logits + per-channel Dice.

This also solves cross-dataset alignment for free -- regions are unions, so the
fact that BraTS 2021 encodes ET as 4 and MU-Glioma-Post encodes it as 3 stops
mattering once both are remapped to canonical integers.
"""

from __future__ import annotations

import numpy as np
import torch
from monai.transforms import MapTransform

from .datasets import REGIONS, REGISTRY


def to_canonical(seg: np.ndarray, dataset: str) -> np.ndarray:
    """Native integers -> canonical {0,1:NETC,2:SNFH,3:ET,4:RC}."""
    if dataset not in REGISTRY:
        raise KeyError(f"no spec for '{dataset}'; add one to datasets.REGISTRY")
    lm = REGISTRY[dataset].label_map
    s = np.rint(np.asarray(seg)).astype(np.int16)
    out = np.zeros_like(s, dtype=np.uint8)
    for src, dst in lm.items():
        if dst:                        # 0 -> 0 is already the fill value
            out[s == src] = dst
    return out


def to_regions(canon: np.ndarray, include_rc: bool = False) -> np.ndarray:
    """Canonical label map -> (C, ...) binary channels, C = 3 or 4."""
    chans = [np.isin(canon, ids) for ids in REGIONS.values()]
    if include_rc:
        chans.append(canon == 4)
    return np.stack(chans).astype(np.uint8)


def regions_from_logits(logits: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """(B,C,...) logits -> binary channels. Threshold, never argmax."""
    return (torch.sigmoid(logits) > thresh)


def regions_to_labelmap(region_bin: np.ndarray) -> np.ndarray:
    """Binary channels -> a single label map, FOR VISUALISATION ONLY.

    Painted outermost-first so the nesting is respected: WT then TC then ET.
    Never use this for metrics -- metrics are per-channel.
    """
    wt, tc, et = region_bin[0], region_bin[1], region_bin[2]
    out = np.zeros(wt.shape, dtype=np.uint8)
    out[wt] = 2          # SNFH
    out[tc] = 1          # NETC
    out[et] = 3          # ET
    if region_bin.shape[0] > 3:
        out[region_bin[3]] = 4
    return out


class ToCanonicalRegionsd(MapTransform):
    """MONAI transform: load native seg -> canonical -> nested sigmoid channels.

    Output has shape (C, H, W, D) with C = 3 (or 4 with RC).
    """

    def __init__(self, keys, dataset: str, include_rc: bool = False,
                 allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.dataset = dataset
        self.include_rc = include_rc

    def __call__(self, data):
        d = dict(data)
        for k in self.key_iterator(d):
            arr = np.asarray(d[k])
            if arr.ndim == 4 and arr.shape[0] == 1:      # drop channel dim if present
                arr = arr[0]
            canon = to_canonical(arr, self.dataset)
            d[k] = to_regions(canon, self.include_rc)
        return d


def assert_nesting(region_bin: np.ndarray, name: str = "") -> None:
    """ET must be inside TC must be inside WT. Cheap invariant, catches
    remap errors that otherwise train silently."""
    wt, tc, et = region_bin[0].astype(bool), region_bin[1].astype(bool), region_bin[2].astype(bool)
    if not (et <= tc).all():
        raise AssertionError(f"{name}: ET not contained in TC -- label map is wrong")
    if not (tc <= wt).all():
        raise AssertionError(f"{name}: TC not contained in WT -- label map is wrong")
