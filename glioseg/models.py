"""Every metric the rubric names, computed PER CHANNEL on nested regions.

Rubric asks for "accuracy, precision, recall, F1 score, IoU, mAP". For binary
overlap F1 IS Dice -- both names are emitted and the identity is stated in the
report, because examiners ask.

Voxel volume always comes from the NIfTI affine, never a hardcoded constant.
"""

from __future__ import annotations

import numpy as np

from .datasets import REGION_NAMES

EPS = 1e-8
VOLUME_BINS = [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 15), (15, 1e9)]


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #

def binary_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    dice = 2 * tp / (2 * tp + fp + fn + EPS)
    return {
        "dice": dice, "f1": dice,                     # identical by definition
        "iou": tp / (tp + fp + fn + EPS),
        "precision": tp / (tp + fp + EPS),
        "recall": tp / (tp + fn + EPS),
        "specificity": tn / (tn + fp + EPS),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + EPS),
        "tp": tp, "fp": fp, "fn": fn,
        "gt_empty": bool(gt.sum() == 0),
        "pred_empty": bool(pred.sum() == 0),
    }


def average_precision(prob: np.ndarray, gt: np.ndarray,
                      thresholds: np.ndarray | None = None) -> float:
    """Area under the PR curve for one region, swept over thresholds.

    This is our reading of the rubric's "mAP" for segmentation. Define it
    explicitly in the report -- it is not a standard BraTS metric.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    gt = np.asarray(gt).astype(bool)
    if gt.sum() == 0:
        return float("nan")
    pts = []
    for t in thresholds:
        p = prob >= t
        if not p.any():
            continue                              # precision undefined; skip
        s = binary_scores(p, gt)
        pts.append((s["recall"], s["precision"]))
    if not pts:
        return 0.0
    pts.sort()
    if pts[0][0] > 0:                             # anchor at recall 0
        pts.insert(0, (0.0, pts[0][1]))
    r = np.array([a for a, _ in pts])
    p = np.array([b for _, b in pts])
    # np.trapz was REMOVED in NumPy 2.x; np.trapezoid is the replacement.
    # Must be checked lazily -- getattr(np, "trapezoid", np.trapz) evaluates
    # the fallback eagerly and raises on NumPy 2.
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(p, r))


def score_case(probs: np.ndarray, gt_channels: np.ndarray,
               thresh: float = 0.5, with_ap: bool = True) -> dict:
    """probs, gt_channels: (C, ...) with C = 3 or 4. Returns a flat dict."""
    out = {}
    names = REGION_NAMES + (["RC"] if probs.shape[0] > 3 else [])
    for i, nm in enumerate(names):
        for k, v in binary_scores(probs[i] >= thresh, gt_channels[i]).items():
            out[f"{nm}_{k}"] = v
        if with_ap:
            out[f"{nm}_AP"] = average_precision(probs[i], gt_channels[i])
    for k in ("dice", "iou", "precision", "recall", "accuracy"):
        vals = [out[f"{n}_{k}"] for n in names if not out[f"{n}_gt_empty"]]
        out[f"mean_{k}"] = float(np.mean(vals)) if vals else float("nan")
    if with_ap:
        aps = [out[f"{n}_AP"] for n in names if not np.isnan(out[f"{n}_AP"])]
        out["mAP"] = float(np.mean(aps)) if aps else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #

def voxel_volume_mm3(affine) -> float:
    return float(abs(np.linalg.det(np.asarray(affine)[:3, :3])))


def stratified_by_volume(records: list[dict], region: str = "ET",
                         bins=VOLUME_BINS) -> list[dict]:
    """Does performance collapse on small lesions? Turns a benchmark into a
    finding, and it is the individual-contribution depth layer."""
    rows = []
    for lo, hi in bins:
        sel = [r for r in records
               if r.get("gt_volume_cm3") is not None and lo <= r["gt_volume_cm3"] < hi]
        if not sel:
            continue
        d = [r[f"{region}_dice"] for r in sel]
        miss = sum(1 for r in sel if r.get(f"{region}_pred_empty"))
        rows.append({"bin_cm3": f"{lo}-{'inf' if hi > 1e8 else hi}",
                     "n": len(sel),
                     "mean_dice": round(float(np.mean(d)), 4),
                     "std_dice": round(float(np.std(d)), 4),
                     "n_missed": miss,
                     "miss_rate": round(miss / len(sel), 4)})
    return rows


# --------------------------------------------------------------------------- #
# Aggregation and statistics
# --------------------------------------------------------------------------- #

def aggregate(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = [k for k, v in records[0].items() if isinstance(v, (int, float, bool))]
    out = {}
    for k in keys:
        v = np.array([float(r[k]) for r in records if r.get(k) is not None], dtype=float)
        v = v[~np.isnan(v)]
        if v.size:
            out[f"{k}_mean"] = round(float(v.mean()), 4)
            out[f"{k}_std"] = round(float(v.std()), 4)
    out["n_cases"] = len(records)
    return out


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return (round(float(np.percentile(means, 100 * alpha / 2)), 4),
            round(float(np.percentile(means, 100 * (1 - alpha / 2))), 4))


def wilcoxon_paired(a, b):
    """Paired Wilcoxon signed-rank. Use on per-case scores from two arms
    evaluated on the SAME cases."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if a.size < 5 or np.allclose(a, b):
        return {"n": int(a.size), "statistic": None, "p": None,
                "note": "too few pairs or identical"}
    s, p = wilcoxon(a, b)
    return {"n": int(a.size), "statistic": float(s), "p": float(p)}


def cohens_d(a, b) -> float:
    """Paired Cohen's d (mean difference / SD of differences)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    return float(d.mean() / (d.std(ddof=1) + EPS)) if d.size > 1 else float("nan")


def fdr_correct(pvals, alpha=0.05):
    """Benjamini-Hochberg. Returns (rejected, adjusted p)."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return (adj <= alpha), np.round(adj, 5)


def compare_arms(records_a: list[dict], records_b: list[dict],
                 key: str = "mean_dice", by: str = "case") -> dict:
    """Paired comparison of two arms on shared cases. The Tier-4 workhorse."""
    ia = {r[by]: r for r in records_a if by in r}
    ib = {r[by]: r for r in records_b if by in r}
    shared = sorted(set(ia) & set(ib))
    a = [ia[c].get(key) for c in shared]
    b = [ib[c].get(key) for c in shared]
    return {"n_shared": len(shared),
            "mean_a": round(float(np.nanmean(a)), 4) if a else None,
            "mean_b": round(float(np.nanmean(b)), 4) if b else None,
            "delta": round(float(np.nanmean(a) - np.nanmean(b)), 4) if a else None,
            "ci_a": bootstrap_ci(a), "ci_b": bootstrap_ci(b),
            "wilcoxon": wilcoxon_paired(a, b), "cohens_d": round(cohens_d(a, b), 4)}
