"""Transforms, splits and loaders.

Two invariants enforced here:
  1. Splits are SUBJECT-level. MU-Glioma-Post has multiple timepoints per
     patient; splitting at timepoint level leaks and inflates every metric.
  2. Labels become nested sigmoid channels (WT/TC/ET), never a class index.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from monai import transforms as T
import torch
from monai.data import DataLoader, Dataset, PersistentDataset
from monai.data.utils import list_data_collate

from .config import Config
from .datasets import REGISTRY, find_cases
from .regions import ToCanonicalRegionsd


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #

def make_split(cases: list[dict], cfg: Config) -> dict:
    """Subject-level split. Deterministic given cfg.seed."""
    subs = sorted({c["subject"] for c in cases})
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(subs)
    n = len(subs)
    n_tr = int(n * cfg.split[0])
    n_va = int(n * cfg.split[1])
    return {"train": subs[:n_tr],
            "val": subs[n_tr:n_tr + n_va],
            "test": subs[n_tr + n_va:]}


def save_split(split: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(split, indent=2))
    return p


def load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def apply_split(cases: list[dict], split: dict) -> dict[str, list[dict]]:
    idx = {s: k for k, v in split.items() for s in v}
    out: dict[str, list[dict]] = {k: [] for k in split}
    for c in cases:
        k = idx.get(c["subject"])
        if k:
            out[k].append(c)
    return out


def subsample_train(parts: dict[str, list[dict]], n: int | None,
                    seed: int) -> dict[str, list[dict]]:
    """Limit the TRAIN split to n cases. Val and test are left untouched.

    This is the Tier-3 data-efficiency knob, and holding val/test fixed is what
    makes it a valid experiment: vary training size, measure on an unchanged
    evaluation set. Changing both at once measures nothing.

    Subsampling happens AFTER the split, never before. Doing it before is a trap
    -- subsample() and make_split() both shuffle with default_rng(seed), so with
    the same seed they produce the same order and the first n subjects are
    exactly the ones assigned to train, leaving val and test EMPTY.
    """
    if n is None or n >= len(parts["train"]):
        return parts
    subs = sorted({c["subject"] for c in parts["train"]})
    # offset the seed so this shuffle cannot coincide with make_split's
    rng = np.random.default_rng(seed + 9973)
    rng.shuffle(subs)
    keep, out = set(), []
    for sub in subs:
        keep.add(sub)
        out = [c for c in parts["train"] if c["subject"] in keep]
        if len(out) >= n:
            break
    return {**parts, "train": out[:n]}


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def build_transforms(cfg: Config, train: bool):
    keys = ["image", "label"]
    # ORDER IS LOAD-BEARING. Reorient while the label is still a single-channel
    # label map WITH its affine metadata; only then expand it into nested region
    # channels. Doing the region conversion first strips the metadata, and
    # Orientationd then silently reorients the image but NOT the label --
    # producing misaligned pairs that train without error and score wrongly.
    base = [
        # image is a LIST of paths -> LoadImaged stacks them into channels in
        # the order given, which is cfg.modalities (canonical bundle order).
        T.LoadImaged(keys=keys, ensure_channel_first=False),
        T.EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        T.EnsureTyped(keys=keys, track_meta=True),
        T.Orientationd(keys=keys, axcodes="RAS"),
        # label is now (1,H,W,D) RAS -> expand to (C,H,W,D) nested channels
        ToCanonicalRegionsd(keys=["label"], dataset=cfg.dataset,
                            include_rc=cfg.include_rc),
        # Re-assert types AFTER the custom transform. Belt and braces: whatever
        # the transform or the PersistentDataset pickle boundary does, the
        # deterministic prefix must end in tensors or the default collate fails
        # with "'numpy.ndarray' object has no attribute 'numel'".
        T.EnsureTyped(keys=keys, track_meta=True),
        T.CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if not train:
        return T.Compose(base)

    return T.Compose(base + [
        T.SpatialPadd(keys=keys, spatial_size=cfg.patch_size),
        # MONAI's RandCropByPosNegLabel assumes a multi-channel label is ONE-HOT
        # and drops channel 0 as background:
        #     if label.shape[0] > 1: label = label[1:]
        # Our channels are NESTED regions [WT, TC, ET], so that silently discards
        # WT and samples on TC only. Cases with edema but no tumour core then
        # report "Num foregrounds 0" and fall back to uniform random crops.
        # Fix: guide cropping with a SINGLE-channel WT mask, which bypasses the
        # one-hot assumption entirely. Dropped again straight after.
        T.CopyItemsd(keys=["label"], times=1, names=["crop_guide"]),
        T.Lambdad(keys=["crop_guide"], func=lambda x: x[0:1]),   # WT channel only
        T.RandCropByPosNegLabeld(
            keys=keys + ["crop_guide"], label_key="crop_guide",
            spatial_size=cfg.patch_size, pos=2, neg=1,
            num_samples=cfg.samples_per_volume, allow_smaller=True,
        ),
        T.DeleteItemsd(keys=["crop_guide"]),
        T.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        T.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        T.RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        T.RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
        T.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
    ])


def safe_collate(batch):
    """MONAI's collate, but numpy arrays are coerced to tensors first.

    The default collate calls .numel() on every element, so a single numpy array
    anywhere in the batch raises. Rather than depend on every transform and the
    PersistentDataset pickle round-trip preserving tensor types, coerce here --
    this is the one place that cannot be bypassed.
    """
    import numpy as _np

    def coerce(x):
        if isinstance(x, _np.ndarray):
            return torch.from_numpy(_np.ascontiguousarray(x))
        if isinstance(x, dict):
            return {k: coerce(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(coerce(v) for v in x)
        return x

    return list_data_collate(coerce(batch))


# Measured on BraTS 2021: a cached item is ~170 MB (4x240x240x155 float32 image
# plus a 3-channel label; CropForegroundd barely shrinks these volumes). So
# PersistentDataset costs roughly 170 MB x n_cases of local disk -- ~149 GB for
# 875 training cases, which exceeds a Colab VM. Cache only for small subsets.
CACHE_MB_PER_CASE = 170


def _ds(records, tf, cfg: Config, tag: str):
    if cfg.cache_dir and tag == "train":
        est_gb = len(records) * CACHE_MB_PER_CASE / 1024
        if est_gb > 40:
            print(f"  [cache] WARNING: ~{est_gb:.0f} GB needed for {len(records)} "
                  f"cases. Colab has ~200 GB total. Set cache_dir=None or reduce "
                  f"n_cases.")
    if cfg.cache_dir:
        # data_hash, NOT hash -- so every arm shares one cache. See Config.data_hash.
        d = Path(cfg.cache_dir) / f"{cfg.dataset}_{tag}_{cfg.data_hash()}"
        d.mkdir(parents=True, exist_ok=True)
        return PersistentDataset(data=records, transform=tf, cache_dir=str(d))
    return Dataset(data=records, transform=tf)


def build_loaders(cfg: Config, split_path: str | Path | None = None,
                  verbose: bool = True):
    """Returns (train_loader, val_loader, test_loader, meta)."""
    spec = REGISTRY[cfg.dataset]
    cases = find_cases(spec, cfg.data_base, modalities=cfg.modalities,
                       verbose=verbose)
    # NOTE: split FIRST on the full set, subsample the train split AFTER.
    if split_path and Path(split_path).exists():
        split = load_split(split_path)
    else:
        split = make_split(cases, cfg)
        if split_path:
            save_split(split, split_path)

    parts = apply_split(cases, split)
    parts = subsample_train(parts, cfg.n_cases, cfg.seed)

    for k in ("train", "val", "test"):
        if not parts[k]:
            raise RuntimeError(
                f"{k} split is EMPTY. Training without validation or test data "
                f"produces NaN metrics. Check n_cases ({cfg.n_cases}) against the "
                f"split file, and delete a stale split if the case set changed.")
    if verbose:
        print(f"[split] subjects {{{', '.join(f'{k}:{len(v)}' for k,v in split.items())}}}"
              f"  cases {{{', '.join(f'{k}:{len(v)}' for k,v in parts.items())}}}")

    tr = _ds(parts["train"], build_transforms(cfg, True), cfg, "train")
    va = _ds(parts["val"], build_transforms(cfg, False), cfg, "val")
    te = _ds(parts["test"], build_transforms(cfg, False), cfg, "test")

    return (
        DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                   pin_memory=True, drop_last=True, persistent_workers=False,
                   collate_fn=safe_collate),
        DataLoader(va, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
                   collate_fn=safe_collate),
        DataLoader(te, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
                   collate_fn=safe_collate),
        {"split": split, "n_cases": len(cases),
         "counts": {k: len(v) for k, v in parts.items()}},
    )
