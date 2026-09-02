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
from monai.data import DataLoader, Dataset, PersistentDataset

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


def subsample(cases: list[dict], n: int | None, seed: int) -> list[dict]:
    """Take n cases, chosen at SUBJECT granularity so timepoints stay together.
    This is the Tier-3 data-efficiency knob."""
    if n is None or n >= len(cases):
        return cases
    subs = sorted({c["subject"] for c in cases})
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    keep, out = set(), []
    for s in subs:
        keep.add(s)
        out = [c for c in cases if c["subject"] in keep]
        if len(out) >= n:
            break
    return out[:n]


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def build_transforms(cfg: Config, train: bool):
    keys = ["image", "label"]
    base = [
        # image is a LIST of paths -> LoadImaged stacks them into channels in
        # the order given, which is cfg.modalities (canonical bundle order).
        T.LoadImaged(keys=keys),
        T.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")
        if False else T.Identityd(keys=["image"]),   # LoadImaged already stacks
        ToCanonicalRegionsd(keys=["label"], dataset=cfg.dataset,
                            include_rc=cfg.include_rc),
        T.EnsureTyped(keys=keys, track_meta=True),
        T.Orientationd(keys=keys, axcodes="RAS"),
        T.CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
        T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if not train:
        return T.Compose(base)

    return T.Compose(base + [
        T.SpatialPadd(keys=keys, spatial_size=cfg.patch_size),
        T.RandCropByPosNegLabeld(
            keys=keys, label_key="label", spatial_size=cfg.patch_size,
            pos=2, neg=1, num_samples=cfg.samples_per_volume, image_key="image",
            allow_smaller=True,
        ),
        T.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        T.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        T.RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        T.RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
        T.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
    ])


def _ds(records, tf, cfg: Config, tag: str):
    if cfg.cache_dir:
        d = Path(cfg.cache_dir) / f"{cfg.dataset}_{tag}_{cfg.hash()}"
        d.mkdir(parents=True, exist_ok=True)
        return PersistentDataset(data=records, transform=tf, cache_dir=str(d))
    return Dataset(data=records, transform=tf)


def build_loaders(cfg: Config, split_path: str | Path | None = None,
                  verbose: bool = True):
    """Returns (train_loader, val_loader, test_loader, meta)."""
    spec = REGISTRY[cfg.dataset]
    cases = find_cases(spec, cfg.data_base, modalities=cfg.modalities,
                       verbose=verbose)
    cases = subsample(cases, cfg.n_cases, cfg.seed)

    if split_path and Path(split_path).exists():
        split = load_split(split_path)
    else:
        split = make_split(cases, cfg)
        if split_path:
            save_split(split, split_path)

    parts = apply_split(cases, split)
    if verbose:
        print(f"[split] subjects {{{', '.join(f'{k}:{len(v)}' for k,v in split.items())}}}"
              f"  cases {{{', '.join(f'{k}:{len(v)}' for k,v in parts.items())}}}")

    tr = _ds(parts["train"], build_transforms(cfg, True), cfg, "train")
    va = _ds(parts["val"], build_transforms(cfg, False), cfg, "val")
    te = _ds(parts["test"], build_transforms(cfg, False), cfg, "test")

    return (
        DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, num_workers=2,
                   pin_memory=True, drop_last=True, persistent_workers=False),
        DataLoader(va, batch_size=1, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(te, batch_size=1, shuffle=False, num_workers=2, pin_memory=True),
        {"split": split, "n_cases": len(cases),
         "counts": {k: len(v) for k, v in parts.items()}},
    )
