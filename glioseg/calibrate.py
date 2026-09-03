"""Per-region decision-threshold calibration.

A fixed sigmoid threshold of 0.5 is a known source of precision loss when a model
is recall-biased. Dice-optimised training produces overconfident, poorly
calibrated outputs (Yeung et al., arXiv:2111.00528), and 0.5 is then simply the
wrong operating point on the precision-recall curve.

METHODOLOGICAL NOTE, and state this in the report: thresholds are selected on the
VALIDATION split and applied unchanged to the untouched TEST split. Tuning on
validation is standard post-processing, not test leakage -- leakage would require
touching test. The test split is evaluated exactly once, at the chosen threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from .config import Config
from .datasets import REGION_NAMES
from .metrics import aggregate, binary_scores, voxel_volume_mm3
from .evaluate import _affine_of

THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)


@torch.no_grad()
def collect_probabilities(model, loader, cfg: Config, device=None, max_cases=None):
    """Run inference once and keep per-case (probs, gt) so a threshold sweep is
    free. Memory: probs are float16 on CPU; ~50 MB per case at BraTS size, so
    cap max_cases if RAM is tight."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    out = []
    for i, batch in enumerate(loader):
        if max_cases and i >= max_cases:
            break
        x = batch["image"].to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=cfg.amp and dev == "cuda"):
            logits = sliding_window_inference(
                x, cfg.patch_size, cfg.sw_batch_size, model,
                overlap=cfg.sw_overlap, mode="gaussian")
        probs = torch.sigmoid(logits.float())[0].cpu().numpy().astype(np.float16)
        gt = batch["label"][0].cpu().numpy().astype(bool)
        mm3 = voxel_volume_mm3(_affine_of(batch))
        c = batch.get("case", ["?"])
        out.append({"probs": probs, "gt": gt, "voxel_mm3": mm3,
                    "case": c[0] if isinstance(c, (list, tuple)) else str(c)})
        if (i + 1) % 25 == 0:
            print(f"    collected {i+1} cases", flush=True)
    print(f"    collected {len(out)} cases total")
    return out


def sweep(cached: list[dict], thresholds=THRESHOLDS) -> list[dict]:
    """Per-region metrics at every threshold. Pure numpy, no GPU."""
    names = REGION_NAMES + (["RC"] if cached[0]["probs"].shape[0] > 3 else [])
    rows = []
    for t in thresholds:
        row = {"threshold": float(t)}
        for ci, nm in enumerate(names):
            per = [binary_scores(c["probs"][ci] >= t, c["gt"][ci])
                   for c in cached if c["gt"][ci].any()]
            if not per:
                continue
            for k in ("dice", "iou", "precision", "recall"):
                row[f"{nm}_{k}"] = float(np.mean([p[k] for p in per]))
            row[f"{nm}_n"] = len(per)
        for k in ("dice", "precision", "recall"):
            vals = [row[f"{n}_{k}"] for n in names if f"{n}_{k}" in row]
            row[f"mean_{k}"] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)
    return rows


def pick_thresholds(val_rows: list[dict], objective: str = "dice") -> dict:
    """Best threshold PER REGION on validation. Regions differ in prevalence and
    contrast, so one global threshold is unlikely to suit all three."""
    names = [n for n in REGION_NAMES + ["RC"] if f"{n}_{objective}" in val_rows[0]]
    best = {}
    for nm in names:
        r = max(val_rows, key=lambda x: x.get(f"{nm}_{objective}", -1))
        best[nm] = {"threshold": r["threshold"],
                    f"val_{objective}": round(r[f"{nm}_{objective}"], 4),
                    "val_precision": round(r.get(f"{nm}_precision", np.nan), 4),
                    "val_recall": round(r.get(f"{nm}_recall", np.nan), 4)}
    return best


def apply_thresholds(cached: list[dict], per_region: dict) -> tuple[dict, list[dict]]:
    """Score the test split at the validation-chosen thresholds."""
    names = REGION_NAMES + (["RC"] if cached[0]["probs"].shape[0] > 3 else [])
    recs = []
    for c in cached:
        rec = {"case": c["case"], "voxel_mm3": c["voxel_mm3"]}
        for ci, nm in enumerate(names):
            t = per_region.get(nm, {}).get("threshold", 0.5)
            for k, v in binary_scores(c["probs"][ci] >= t, c["gt"][ci]).items():
                rec[f"{nm}_{k}"] = v
            rec[f"{nm}_threshold"] = t
        for k in ("dice", "iou", "precision", "recall", "accuracy"):
            vals = [rec[f"{n}_{k}"] for n in names if not rec[f"{n}_gt_empty"]]
            rec[f"mean_{k}"] = float(np.mean(vals)) if vals else np.nan
        et = names.index("ET")
        rec["gt_volume_cm3"] = float(c["gt"][et].sum() * c["voxel_mm3"] / 1000.0)
        recs.append(rec)
    return aggregate(recs), recs


def calibrate(model, val_loader, test_loader, cfg: Config,
              objective: str = "dice", out_dir=None, verbose=True) -> dict:
    """Full procedure: sweep on val, pick, apply once to test, report the delta."""
    if verbose:
        print("  [1/4] collecting validation probabilities...")
    val_cached = collect_probabilities(model, val_loader, cfg)

    if verbose:
        print("  [2/4] sweeping thresholds on validation...")
    val_rows = sweep(val_cached)
    chosen = pick_thresholds(val_rows, objective)
    if verbose:
        for nm, d in chosen.items():
            print(f"      {nm}: t={d['threshold']:.2f}  val {objective}="
                  f"{d[f'val_{objective}']:.4f}  P={d['val_precision']:.4f}  "
                  f"R={d['val_recall']:.4f}")
    del val_cached

    if verbose:
        print("  [3/4] collecting test probabilities...")
    test_cached = collect_probabilities(model, test_loader, cfg)

    if verbose:
        print("  [4/4] scoring test at 0.5 and at calibrated thresholds...")
    base_agg, base_recs = apply_thresholds(test_cached, {n: {"threshold": 0.5} for n in chosen})
    cal_agg, cal_recs = apply_thresholds(test_cached, chosen)

    delta = {k: round(cal_agg[k] - base_agg[k], 4)
             for k in cal_agg if k.endswith("_mean") and k in base_agg
             and isinstance(cal_agg[k], (int, float))}

    res = {"chosen": chosen, "objective": objective,
           "baseline_0.5": base_agg, "calibrated": cal_agg, "delta": delta,
           "val_sweep": val_rows}
    if verbose:
        print("\n  --- test-set effect of calibration ---")
        for k in ("mean_dice_mean", "mean_precision_mean", "mean_recall_mean",
                  "ET_dice_mean", "WT_dice_mean", "TC_dice_mean"):
            if k in base_agg:
                print(f"      {k:<24} {base_agg[k]:.4f} -> {cal_agg[k]:.4f}"
                      f"  ({delta.get(k, 0):+.4f})")

    if out_dir:
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        (d / "calibration.json").write_text(json.dumps(res, indent=2, default=str))
        (d / "per_case_calibrated.json").write_text(json.dumps(cal_recs, indent=2, default=str))
        if verbose:
            print(f"\n  saved -> {d/'calibration.json'}")
    return res


def threshold_by_volume(cached: list[dict], thresholds=THRESHOLDS,
                        bins=((0, 2), (2, 5), (5, 15), (15, 1e9))) -> list[dict]:
    """Does the optimal threshold depend on lesion size?

    If small lesions prefer a lower threshold and large ones a higher one, then
    no single global threshold can serve both -- which is a second finding, and
    an argument for size-aware post-processing.
    """
    et = REGION_NAMES.index("ET")
    rows = []
    for lo, hi in bins:
        sel = [c for c in cached
               if lo <= c["gt"][et].sum() * c["voxel_mm3"] / 1000.0 < hi
               and c["gt"][et].any()]
        if not sel:
            continue
        best_t, best_d = None, -1.0
        curve = []
        for t in thresholds:
            d = float(np.mean([binary_scores(c["probs"][et] >= t, c["gt"][et])["dice"]
                               for c in sel]))
            curve.append((float(t), round(d, 4)))
            if d > best_d:
                best_t, best_d = float(t), d
        rows.append({"bin_cm3": f"{lo}-{'inf' if hi > 1e8 else hi}", "n": len(sel),
                     "best_threshold": best_t, "best_ET_dice": round(best_d, 4),
                     "dice_at_0.5": round(dict(curve)[0.5], 4) if 0.5 in dict(curve) else None,
                     "curve": curve})
    return rows
