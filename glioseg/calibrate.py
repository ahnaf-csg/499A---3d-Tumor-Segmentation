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


def per_case_optimal_threshold(cached: list[dict], thresholds=THRESHOLDS,
                               region: str = "ET") -> list[dict]:
    """Best threshold for EACH case, with its lesion volume.

    This is the binning-free alternative to threshold_by_volume. Binning throws
    away information and produces cells with n=1 or 2 that support no claim; a
    per-case value lets you regress threshold on log-volume and use every case.
    """
    ci = REGION_NAMES.index(region)
    out = []
    for c in cached:
        if not c["gt"][ci].any():
            continue
        vol = float(c["gt"][ci].sum() * c["voxel_mm3"] / 1000.0)
        curve = [(float(t), binary_scores(c["probs"][ci] >= t, c["gt"][ci])["dice"])
                 for t in thresholds]
        best_t, best_d = max(curve, key=lambda x: x[1])
        d50 = dict(curve).get(0.5, float("nan"))
        out.append({"case": c["case"], "volume_cm3": vol,
                    "log10_volume": float(np.log10(max(vol, 1e-3))),
                    "best_threshold": best_t, "best_dice": round(best_d, 4),
                    "dice_at_0.5": round(d50, 4),
                    "gain": round(best_d - d50, 4)})
    return out


def threshold_volume_regression(per_case: list[dict], n_boot: int = 2000,
                                seed: int = 0) -> dict:
    """Does the optimal threshold depend on lesion size? Continuous test.

    Regresses per-case optimal threshold on log10 lesion volume. Uses every case
    with a non-empty region, so no bin is ever too thin to interpret. A slope
    whose CI excludes zero means no single global threshold serves all sizes.
    """
    if len(per_case) < 8:
        return {"n": len(per_case), "note": "too few cases to regress"}
    x = np.array([r["log10_volume"] for r in per_case])
    y = np.array([r["best_threshold"] for r in per_case])

    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])

    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) > 1:
            slopes.append(np.polyfit(x[i], y[i], 1)[0])
    lo, hi = np.percentile(slopes, [2.5, 97.5]) if slopes else (np.nan, np.nan)

    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(x, y)
    except ImportError:
        rho, pval = float("nan"), float("nan")

    return {"n": len(per_case),
            "slope_per_decade": round(float(slope), 4),
            "slope_95ci": (round(float(lo), 4), round(float(hi), 4)),
            "significant": bool(not (lo <= 0 <= hi)),
            "pearson_r": round(r, 4), "r_squared": round(r**2, 4),
            "spearman_rho": round(float(rho), 4), "spearman_p": round(float(pval), 6),
            "interpretation": (
                "optimal threshold varies with lesion size -- no single global "
                "threshold serves all sizes"
                if not (lo <= 0 <= hi) else
                "no detectable size dependence; a global threshold is adequate")}


def threshold_by_volume(cached: list[dict], thresholds=THRESHOLDS,
                        bins=((0, 2), (2, 5), (5, 15), (15, 1e9)),
                        n_boot: int = 1000, min_n: int = 8, seed: int = 0) -> list[dict]:
    """Does the optimal threshold depend on lesion size?

    If small lesions prefer a lower threshold and large ones a higher one, no
    single global threshold serves both -- which argues for size-aware
    post-processing.

    BINNING MATTERS. On BraTS the six-bin scheme used for stratified Dice leaves
    n = 12/2/6/13/44/112, so the middle bins cannot support a threshold estimate.
    The default here merges to four bins (n ~ 20/13/44/112), and the smallest is
    still reported with a bootstrap interval so the reader can judge it.

    A point estimate of "the best threshold" is meaningless without a spread:
    the Dice-vs-threshold curve is flat near its peak, so the argmax jumps
    between adjacent grid values on resampling. We therefore bootstrap over
    cases within each bin and report the DISTRIBUTION of optimal thresholds.
    """
    et = REGION_NAMES.index("ET")
    rng = np.random.default_rng(seed)
    rows = []

    for lo, hi in bins:
        sel = [c for c in cached
               if lo <= c["gt"][et].sum() * c["voxel_mm3"] / 1000.0 < hi
               and c["gt"][et].any()]
        if not sel:
            continue

        # per-case Dice at every threshold, computed once and reused by the bootstrap
        per_case = np.array([[binary_scores(c["probs"][et] >= t, c["gt"][et])["dice"]
                              for t in thresholds] for c in sel])   # (n_cases, n_thr)
        mean_curve = per_case.mean(axis=0)
        best_i = int(mean_curve.argmax())

        row = {"bin_cm3": f"{lo}-{'inf' if hi > 1e8 else hi}", "n": len(sel),
               "best_threshold": float(thresholds[best_i]),
               "best_ET_dice": round(float(mean_curve[best_i]), 4),
               "dice_at_0.5": round(float(mean_curve[list(thresholds).index(0.5)]), 4)
                              if 0.5 in list(thresholds) else None,
               "gain_over_0.5": None,
               "curve": [(float(t), round(float(d), 4))
                         for t, d in zip(thresholds, mean_curve)]}
        if row["dice_at_0.5"] is not None:
            row["gain_over_0.5"] = round(row["best_ET_dice"] - row["dice_at_0.5"], 4)

        if len(sel) >= min_n:
            # bootstrap the argmax so the point estimate carries a spread
            boots = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(sel), len(sel))
                boots.append(float(thresholds[int(per_case[idx].mean(axis=0).argmax())]))
            row["threshold_95ci"] = (round(float(np.percentile(boots, 2.5)), 2),
                                     round(float(np.percentile(boots, 97.5)), 2))
            row["threshold_iqr"] = (round(float(np.percentile(boots, 25)), 2),
                                    round(float(np.percentile(boots, 75)), 2))
            # how flat is the peak? if many thresholds are within 1% of the best,
            # the argmax is not identifiable and should not be over-read
            near = int((mean_curve >= mean_curve[best_i] - 0.01).sum())
            row["n_thresholds_within_1pct"] = near
        else:
            row["threshold_95ci"] = None
            row["note"] = f"n={len(sel)} < {min_n}: point estimate only, do not interpret"
        rows.append(row)
    return rows


def thresholds_differ_by_size(rows: list[dict]) -> dict:
    """Do two size bins genuinely prefer different thresholds?

    Answered by CI overlap, not by comparing point estimates. Returns the
    verdict so the report can state it rather than implying it from a table.
    """
    usable = [r for r in rows if r.get("threshold_95ci")]
    if len(usable) < 2:
        return {"verdict": "insufficient data", "n_usable_bins": len(usable)}

    small, large = usable[0], usable[-1]
    (sl, sh), (ll, lh) = small["threshold_95ci"], large["threshold_95ci"]
    overlap = not (sh < ll or lh < sl)
    return {
        "smallest_bin": small["bin_cm3"], "largest_bin": large["bin_cm3"],
        "smallest_threshold": small["best_threshold"], "smallest_95ci": (sl, sh),
        "largest_threshold": large["best_threshold"], "largest_95ci": (ll, lh),
        "cis_overlap": overlap,
        "verdict": ("no evidence that optimal threshold depends on lesion size "
                    "(CIs overlap)" if overlap else
                    "optimal threshold DOES depend on lesion size (CIs disjoint) "
                    "-- no single global threshold serves both"),
    }
