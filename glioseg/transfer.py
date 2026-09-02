"""Partial checkpoint loading for the transfer experiment.

Never load silently. A checkpoint whose keys mostly do not match will train
happily and produce meaningless "pretrained" numbers, so this reports matched
and skipped keys and refuses quietly-bad loads by surfacing match_frac.

Known pre-operative BraTS sources (both are 4-in / 3-out sigmoid WT/TC/ET,
which is exactly our representation -- no head surgery needed):
  SegResNet  MONAI Model Zoo bundle `brats_mri_segmentation` (BraTS 2018)
  SwinUNETR  MONAI research-contributions BRATS21 fold checkpoints
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

PREFIXES = ("module.", "model.", "network.", "net.", "_orig_mod.")

BUNDLE_URLS = {
    "segresnet_brats2018":
        "https://huggingface.co/MONAI/brats_mri_segmentation/resolve/main/models/model.pt",
    # SwinUNETR BRATS21 folds live in Project-MONAI/research-contributions;
    # the exact release asset path changes, so fetch it manually and pass the file.
}


def _strip(k: str) -> str:
    for p in PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k


def extract_state_dict(obj) -> dict:
    """Unwrap the many shapes checkpoints arrive in."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "net", "student", "weights"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if any(hasattr(v, "shape") for v in obj.values()):
            return obj
    raise ValueError("could not locate a state_dict in this checkpoint")


def load_pretrained(model: nn.Module, ckpt: str | Path,
                    verbose: bool = True, min_frac: float = 0.30) -> dict:
    p = Path(ckpt)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")

    raw = torch.load(str(p), map_location="cpu", weights_only=False)
    sd = {_strip(k): v for k, v in extract_state_dict(raw).items()}
    tgt = model.state_dict()

    matched, shape_mismatch, missing = {}, [], []
    for k, v in tgt.items():
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            matched[k] = sd[k]
        elif k in sd:
            shape_mismatch.append((k, tuple(sd[k].shape), tuple(v.shape)))
        else:
            missing.append(k)
    unexpected = [k for k in sd if k not in tgt]

    model.load_state_dict(matched, strict=False)
    frac = len(matched) / max(len(tgt), 1)

    info = {"checkpoint": str(p), "match_frac": round(frac, 4),
            "matched": len(matched), "target_tensors": len(tgt),
            "shape_mismatch": shape_mismatch[:8],
            "missing": missing[:8], "unexpected": unexpected[:8]}

    if verbose:
        print(f"  [pretrained] {p.name}: matched {len(matched)}/{len(tgt)} "
              f"({frac:.1%})")
        if shape_mismatch:
            print(f"    shape mismatches ({len(shape_mismatch)}), first few:")
            for k, a, b in shape_mismatch[:4]:
                print(f"      {k}: ckpt {a} vs model {b}")
        if frac < min_frac:
            print(f"    WARNING: only {frac:.1%} matched. This is probably the "
                  f"wrong architecture or a different feature_size. Do NOT report "
                  f"this as a pretrained result.")
        elif shape_mismatch:
            print("    Head mismatch is expected if out_channels differ; body "
                  "match is what matters.")
    return info


def freeze_encoder(model: nn.Module, patterns=("encoder", "down", "swinViT", "stages")) -> int:
    """Optional: freeze the encoder for a linear-probe style fine-tune.
    Returns the number of frozen tensors."""
    n = 0
    for name, prm in model.named_parameters():
        if any(pat in name for pat in patterns):
            prm.requires_grad_(False)
            n += 1
    return n
