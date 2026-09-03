"""RANO response classification from segmented enhancing-tumour volumes.

CONSTRAINT WORTH UNDERSTANDING: RANO is LONGITUDINAL. It classifies response by
comparing two timepoints, so it cannot be computed on a cross-sectional dataset
like BraTS 2021 (one scan per patient). This module therefore only applies to
SAILOR or MU-Glioma-Post.

Pipeline:  segmentation -> ET volume per session -> dV% per consecutive pair
           -> RANO category -> agreement (Cohen's kappa) against a reference

BINARY, NOT 4-CLASS. Progression-versus-not is far more robust here. On this
team's own SAILOR data, automated-vs-expert agreement was kappa 0.707 for the
binary RANO category but Lin's CCC 0.052 for continuous volume change. TRACE
(arXiv:2606.30313) likewise reports 4-class macro F1 0.4769 against 0.7085
binary on LUMIERE. Four-class boundaries also require CR/PR/SD definitions that
should be read from Wen et al. before being reported.

Reference: Wen et al., "RANO 2.0", J Clin Oncol 2023;41(33):5187-5199,
DOI 10.1200/JCO.23.01059. Volumetric criterion used here: >= 40% ET volume
increase = progressive disease.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PD_THRESHOLD = 0.40          # >= +40% ET volume => progressive disease (RANO 2.0)
RANO_MEASURABLE_CM3 = 0.5    # ~10mm x 10mm bidimensional; below this, unmeasurable


def volumes_from_masks(records: list[dict], region_index: int = 2) -> dict:
    """[{'case','subject','session',...}] -> {subject: [(session, volume_cm3)]}

    `region_index` 2 = ET in the canonical [WT, TC, ET] channel order.
    """
    per = defaultdict(list)
    for r in records:
        sess = r.get("session")
        if sess is None:
            m = re.search(r"ses-?(\d+)|Timepoint_?(\d+)", r.get("case", ""))
            sess = int(next(g for g in m.groups() if g)) if m else 0
        per[r["subject"]].append((int(sess), float(r["volume_cm3"])))
    return {k: sorted(v) for k, v in per.items()}


def delta_pairs(vols: dict, intervals: dict | None = None,
                min_baseline_cm3: float = 0.0) -> list[dict]:
    """Consecutive session pairs with dV%.

    Pairs whose baseline volume is 0 are dropped -- dV% divides by it. That is a
    real data loss to report, not a silent filter: on SAILOR it removed 37 of
    243 pairs (15.2%).
    """
    out, dropped = [], 0
    for sub, seq in vols.items():
        for i in range(len(seq) - 1):
            (s0, v0), (s1, v1) = seq[i], seq[i + 1]
            if v0 <= min_baseline_cm3:
                dropped += 1
                continue
            gap = None
            if intervals and sub in intervals and i < len(intervals[sub]):
                gap = intervals[sub][i]
            out.append({"subject": sub, "session_a": s0, "session_b": s1,
                        "vol_a_cm3": v0, "vol_b_cm3": v1,
                        "delta_pct": (v1 - v0) / v0 * 100.0,
                        "interval_days": gap,
                        "baseline_measurable": v0 >= RANO_MEASURABLE_CM3})
    if dropped:
        print(f"  [rano] dropped {dropped} pair(s) with zero baseline volume "
              f"({dropped/(len(out)+dropped)*100:.1f}%)")
    return out


def rano_binary(delta_pct: float, threshold: float = PD_THRESHOLD) -> int:
    """1 = progressive disease, 0 = not. RANO 2.0 volumetric criterion."""
    return int(delta_pct >= threshold * 100.0)


def classify(pairs: list[dict], threshold: float = PD_THRESHOLD) -> list[dict]:
    for p in pairs:
        p["rano_pd"] = rano_binary(p["delta_pct"], threshold)
    return pairs


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #

def cohens_kappa(a, b) -> dict:
    """Binary Cohen's kappa with the full confusion matrix."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = a.size
    if n == 0:
        return {"kappa": float("nan"), "n": 0}
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    po = (n11 + n00) / n
    pe = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / n**2
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"kappa": round(float(kappa), 4), "raw_agreement": round(po, 4),
            "n": n, "confusion": {"both_PD": n11, "a_only": n10,
                                  "b_only": n01, "neither": n00}}


def kappa_ci_clustered(pairs: list[dict], key_a: str, key_b: str,
                       n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap CI resampling SUBJECTS, not pairs.

    Pairs from one patient are correlated. Resampling pairs would understate the
    variance and give a CI that is too narrow.
    """
    by_sub = defaultdict(list)
    for p in pairs:
        by_sub[p["subject"]].append(p)
    subs = list(by_sub)
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        sel = [p for s in pick for p in by_sub[s]]
        k = cohens_kappa([p[key_a] for p in sel], [p[key_b] for p in sel])["kappa"]
        if not np.isnan(k):
            ks.append(k)
    if not ks:
        return (float("nan"), float("nan"))
    return (round(float(np.percentile(ks, 100 * alpha / 2)), 4),
            round(float(np.percentile(ks, 100 * (1 - alpha / 2))), 4))


def compare_to_reference(model_records: list[dict], ref_records: list[dict],
                         intervals: dict | None = None,
                         threshold: float = PD_THRESHOLD,
                         out_dir=None, verbose=True) -> dict:
    """Full evaluation: model-derived RANO vs reference-derived RANO.

    Both record lists need 'subject', 'case'/'session' and 'volume_cm3'.
    """
    mv = classify(delta_pairs(volumes_from_masks(model_records), intervals), threshold)
    rv = classify(delta_pairs(volumes_from_masks(ref_records), intervals), threshold)

    idx = {(p["subject"], p["session_a"], p["session_b"]): p for p in rv}
    merged = []
    for p in mv:
        k = (p["subject"], p["session_a"], p["session_b"])
        if k in idx:
            merged.append({**p, "ref_pd": idx[k]["rano_pd"],
                           "ref_delta_pct": idx[k]["delta_pct"],
                           "ref_vol_a_cm3": idx[k]["vol_a_cm3"]})

    if not merged:
        return {"error": "no overlapping pairs between model and reference"}

    ag = cohens_kappa([p["rano_pd"] for p in merged], [p["ref_pd"] for p in merged])
    ci = kappa_ci_clustered(merged, "rano_pd", "ref_pd")

    # continuous agreement, for the contrast that motivated going categorical
    md = np.array([p["delta_pct"] for p in merged])
    rd = np.array([p["ref_delta_pct"] for p in merged])
    ccc = (2 * np.corrcoef(md, rd)[0, 1] * md.std() * rd.std()) / (
        md.var() + rd.var() + (md.mean() - rd.mean()) ** 2)

    meas = [p for p in merged if p["ref_vol_a_cm3"] >= RANO_MEASURABLE_CM3]
    ag_meas = cohens_kappa([p["rano_pd"] for p in meas],
                           [p["ref_pd"] for p in meas]) if meas else {}

    res = {"n_pairs": len(merged), "n_subjects": len({p["subject"] for p in merged}),
           "pd_threshold_pct": threshold * 100,
           "categorical": {**ag, "kappa_95ci": ci},
           "continuous_ccc_delta_pct": round(float(ccc), 4),
           "measurable_only": {**ag_meas, "n_pairs": len(meas)}}

    if verbose:
        print(f"\n  RANO binary agreement over {res['n_pairs']} pairs "
              f"({res['n_subjects']} subjects)")
        print(f"    Cohen's kappa      {ag['kappa']}  95% CI {ci}  "
              f"(subject-clustered bootstrap)")
        print(f"    raw agreement      {ag['raw_agreement']}")
        print(f"    confusion          {ag['confusion']}")
        print(f"    continuous CCC     {res['continuous_ccc_delta_pct']}  "
              f"<- contrast with the categorical result")
        if meas:
            print(f"    RANO-measurable subset (baseline >= {RANO_MEASURABLE_CM3} cm3): "
                  f"kappa {ag_meas['kappa']}, n={len(meas)}")
        print("\n    CEILINGS to report alongside: automated-vs-expert on identical")
        print("    images gives kappa 0.707 and Dice 0.534. The model cannot")
        print("    exceed the agreement between two annotations of the same data.")

    if out_dir:
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        (d / "rano.json").write_text(json.dumps(res, indent=2, default=str))
        (d / "rano_pairs.json").write_text(json.dumps(merged, indent=2, default=str))
        if verbose:
            print(f"\n  saved -> {d/'rano.json'}")
    return res


def load_intervals(base: str | Path, pattern="**/intervals-days.txt") -> dict:
    """SAILOR ships one integer per line per patient, one per inter-session gap."""
    out = {}
    for p in Path(base).glob(pattern):
        m = re.search(r"(sub-\d+)", str(p))
        if m:
            out[m.group(1)] = [int(x) for x in re.findall(r"\d+", p.read_text())]
    return out


# --------------------------------------------------------------------------- #
# Volume extraction
# --------------------------------------------------------------------------- #

def volumes_from_reference(spec_name: str, base, region_index: int = 2,
                           verbose: bool = True) -> list[dict]:
    """ET volumes straight from the dataset's own segmentations (the reference).

    No model involved -- this is the ground-truth arm of the comparison.
    """
    import nibabel as nib
    from .datasets import REGISTRY, find_cases
    from .regions import to_canonical, to_regions
    from .metrics import voxel_volume_mm3

    spec = REGISTRY[spec_name]
    cases = find_cases(spec, base, verbose=False)
    out = []
    for i, rec in enumerate(cases):
        img = nib.load(rec["label"])
        arr = np.asarray(img.dataobj)
        reg = to_regions(to_canonical(arr, spec_name))
        mm3 = voxel_volume_mm3(img.affine)
        out.append({"case": rec["case"], "subject": rec["subject"],
                    "volume_cm3": float(reg[region_index].sum() * mm3 / 1000.0)})
        if verbose and (i + 1) % 50 == 0:
            print(f"    reference volumes: {i+1}/{len(cases)}", flush=True)
    if verbose:
        v = np.array([r["volume_cm3"] for r in out])
        print(f"  reference: {len(out)} sessions, {len({r['subject'] for r in out})} "
              f"subjects, ET cm3 median {np.median(v):.2f} "
              f"(min {v.min():.2f}, max {v.max():.2f}), {int((v==0).sum())} zero")
    return out


@torch.no_grad()
def volumes_from_model(model, spec_name: str, base, cfg, region_index: int = 2,
                       thresholds: dict | None = None, verbose: bool = True) -> list[dict]:
    """ET volumes from MODEL predictions. Inference only, no training.

    `thresholds` accepts the per-region dict from calibrate.pick_thresholds so
    the calibrated operating point carries over to this experiment.
    """
    import torch as _t
    from monai.inferers import sliding_window_inference
    from .datasets import REGISTRY, find_cases
    from .data import build_transforms
    from .metrics import voxel_volume_mm3
    from .evaluate import _affine_of

    dev = "cuda" if _t.cuda.is_available() else "cpu"
    model.eval()
    spec = REGISTRY[spec_name]
    saved_ds, cfg.dataset = cfg.dataset, spec_name        # transforms are per-dataset
    tf = build_transforms(cfg, train=False)
    cases = find_cases(spec, base, verbose=verbose)

    names = ["WT", "TC", "ET"]
    thr = (thresholds or {}).get(names[region_index], {}).get("threshold", 0.5)
    if verbose:
        print(f"  inference on {spec_name} at threshold {thr:.2f} "
              f"({'calibrated' if thresholds else 'default'})")

    out = []
    for i, rec in enumerate(cases):
        d = tf(rec)
        x = d["image"].unsqueeze(0).to(dev)
        with _t.autocast(dev, dtype=_t.float16, enabled=cfg.amp and dev == "cuda"):
            logits = sliding_window_inference(x, cfg.patch_size, cfg.sw_batch_size,
                                              model, overlap=cfg.sw_overlap,
                                              mode="gaussian")
        prob = _t.sigmoid(logits.float())[0, region_index].cpu().numpy()
        mm3 = voxel_volume_mm3(_affine_of(d))
        out.append({"case": rec["case"], "subject": rec["subject"],
                    "volume_cm3": float((prob >= thr).sum() * mm3 / 1000.0)})
        if verbose and (i + 1) % 25 == 0:
            print(f"    model volumes: {i+1}/{len(cases)}", flush=True)

    cfg.dataset = saved_ds
    if verbose:
        v = np.array([r["volume_cm3"] for r in out])
        print(f"  model: {len(out)} sessions, ET cm3 median {np.median(v):.2f} "
              f"(min {v.min():.2f}, max {v.max():.2f}), {int((v==0).sum())} zero")
    return out
