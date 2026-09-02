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

from .datasets import REGIONS, REGISTRY

# torch and monai are imported LAZILY below. Tier-0 verification needs only
# numpy + nibabel, so a broken or missing MONAI must not block it.


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


def regions_from_logits(logits, thresh: float = 0.5):
    """(B,C,...) logits -> binary channels. Threshold, never argmax."""
    import torch
    return torch.sigmoid(logits) > thresh


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


def _make_transform_class():
    """Build the MONAI transform only when asked, so importing this module
    never requires MONAI."""
    from monai.transforms import MapTransform

    class _ToCanonicalRegionsd(MapTransform):
        """Load native seg -> canonical integers -> nested sigmoid channels.
        Output shape (C, H, W, D) with C = 3, or 4 when include_rc."""

        def __init__(self, keys, dataset: str, include_rc: bool = False,
                     allow_missing_keys: bool = False):
            super().__init__(keys, allow_missing_keys)
            self.dataset = dataset
            self.include_rc = include_rc

        def __call__(self, data):
            import torch
            d = dict(data)
            for k in self.key_iterator(d):
                src = d[k]
                arr = src.cpu().numpy() if hasattr(src, "cpu") else np.asarray(src)
                if arr.ndim == 4 and arr.shape[0] == 1:   # (1,H,W,D) -> (H,W,D)
                    arr = arr[0]
                out = to_regions(to_canonical(arr, self.dataset), self.include_rc)
                out = torch.from_numpy(np.ascontiguousarray(out)).to(torch.uint8)
                # Preserve the MetaTensor wrapper so affine/meta survive. Losing
                # it breaks downstream spatial transforms AND makes the default
                # collate fail on numpy ("no attribute 'numel'").
                if hasattr(src, "meta"):
                    from monai.data import MetaTensor
                    out = MetaTensor(out, meta=dict(src.meta),
                                     applied_operations=list(
                                         getattr(src, "applied_operations", [])))
                d[k] = out
            return d

    return _ToCanonicalRegionsd


def ToCanonicalRegionsd(keys, dataset: str, include_rc: bool = False,
                        allow_missing_keys: bool = False):
    """Factory with the same call signature as a MONAI transform class."""
    return _make_transform_class()(keys, dataset, include_rc, allow_missing_keys)


def assert_nesting(region_bin: np.ndarray, name: str = "") -> None:
    """ET must be inside TC must be inside WT. Cheap invariant, catches
    remap errors that otherwise train silently."""
    wt, tc, et = region_bin[0].astype(bool), region_bin[1].astype(bool), region_bin[2].astype(bool)
    if not (et <= tc).all():
        raise AssertionError(f"{name}: ET not contained in TC -- label map is wrong")
    if not (tc <= wt).all():
        raise AssertionError(f"{name}: TC not contained in WT -- label map is wrong")
