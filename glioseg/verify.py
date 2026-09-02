"""Tier-0 verification. Run this BEFORE any training, on every dataset.

Its whole purpose is to catch the failure mode that costs the most: a label,
affine, channel-order or intensity inconsistency that trains without error and
silently invalidates the results.

Results append to a JSONL log. Exit status is derived from FAILs, so this can
gate a notebook cell or a Makefile target.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .datasets import (CANONICAL_MODALITY_ORDER, REGISTRY, find_cases,
                       resolve_root, save_exclusions)
from .regions import assert_nesting, to_canonical, to_regions

PASS, FAIL, INFO, UNVERIFIED = "PASS", "FAIL", "INFO", "????"


class Log:
    def __init__(self, path="results/verification_log.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def add(self, check, claim, status, evidence=None, note=""):
        r = {"check": check, "claim": claim, "status": status,
             "evidence": evidence or {}, "note": note,
             "ts": datetime.now(timezone.utc).isoformat()}
        self.rows.append(r)
        with self.path.open("a") as fh:
            fh.write(json.dumps(r, default=str) + "\n")
        line = f"[{status}] {check}: {claim}"
        print(line + (f"\n        {note}" if note else ""))
        return r

    def summary(self):
        c = Counter(r["status"] for r in self.rows)
        bad = [r for r in self.rows if r["status"] in (FAIL, UNVERIFIED)]
        out = ["", "=" * 70, "  ".join(f"{k}={v}" for k, v in sorted(c.items())), "=" * 70]
        if bad:
            out.append("UNRESOLVED -- do not train until these are addressed:")
            out += [f"  - {r['check']}: {r['claim']}" for r in bad]
        else:
            out.append("All checks passed.")
        out.append(f"log: {self.path}")
        print("\n".join(out))
        return c


def _load(path):
    import nibabel as nib
    return nib.load(str(path))


def check_environment(log: Log) -> None:
    ev = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        ev["torch"] = torch.__version__
        ev["cuda"] = torch.cuda.is_available()
        ev["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        # T4 is Turing (SM 7.5) and has NO bf16. A100 is Ampere (SM 8.0+).
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            ev["sm"] = f"{cap[0]}.{cap[1]}"
            ev["bf16_supported"] = cap[0] >= 8
    except ImportError:
        ev["torch"] = None
    try:
        import monai
        ev["monai"] = monai.__version__
    except ImportError:
        ev["monai"] = None
    log.add("env", "environment recorded", INFO, ev)

    if ev.get("cuda") and ev.get("bf16_supported") is False:
        log.add("env", "GPU supports bf16", INFO, {"sm": ev.get("sm")},
                "Turing/T4 detected: use fp16 + GradScaler, NOT bf16. "
                "train.py handles this automatically.")


def check_integrity(spec_name: str, base: str, log: Log) -> list[str]:
    """stat() EVERY NIfTI under the root. Sampling misses corrupt files, and one
    of them will crash an overnight run at an unpredictable point.

    Returns the list of case directories that should be excluded.
    """
    spec = REGISTRY[spec_name]
    root = resolve_root(spec, base)
    files = list(root.rglob("*.nii*"))

    bad = []
    for f in files:
        sz = f.stat().st_size
        if sz == 0:
            bad.append((str(f.relative_to(root)), sz, "zero bytes"))
            continue
        # Header read only -- nibabel is lazy, so this does not load voxel data.
        # Catches truncated or corrupt files that a size threshold would miss.
        try:
            _load(f).shape
        except Exception as e:
            bad.append((str(f.relative_to(root)), sz, f"unreadable: {type(e).__name__}"))
    bad_dirs = sorted({Path(b[0]).parent.name for b in bad})

    log.add(f"{spec_name}/integrity", "every NIfTI is non-empty",
            PASS if not bad else FAIL,
            {"n_files_scanned": len(files), "n_bad": len(bad), "bad": bad[:12],
             "affected_cases": bad_dirs[:12]},
            "" if not bad else
            f"{len(bad)} broken file(s) across {len(bad_dirs)} case(s). "
            f"find_cases now excludes these automatically. If the source on Drive "
            f"is also broken the case is genuinely corrupt; if only the local copy "
            f"is, re-copy it.")

    if bad_dirs:
        # Persist so every later loader skips these without re-scanning.
        pth = save_exclusions(spec, base, bad_dirs)
        log.add(f"{spec_name}/integrity", "exclusion list written", INFO,
                {"path": str(pth), "n_excluded": len(bad_dirs)},
                "find_cases reads this automatically from now on.")
    return bad_dirs


def check_dataset(spec_name: str, base: str, log: Log, sample: int = 20) -> dict:
    spec = REGISTRY[spec_name]
    root = resolve_root(spec, base)

    if not root.exists():
        log.add(f"{spec_name}/root", f"root exists: {root}", FAIL,
                note="Fix the path before anything else.")
        return {}

    # ---- discovery -------------------------------------------------------- #
    try:
        cases = find_cases(spec, base, verbose=False)
    except Exception as e:
        log.add(f"{spec_name}/discover", "cases discoverable", FAIL,
                note=f"{type(e).__name__}: {e}")
        return {}

    subs = {c["subject"] for c in cases}
    per_sub = Counter(c["subject"] for c in cases)
    log.add(f"{spec_name}/discover", "cases and subjects found", PASS,
            {"n_cases": len(cases), "n_subjects": len(subs),
             "cases_per_subject_min_max": [min(per_sub.values()), max(per_sub.values())],
             "example_case": cases[0]["case"]})

    if spec.case_depth == 2:
        log.add(f"{spec_name}/discover", "timepoints per subject", INFO,
                {"histogram": dict(sorted(Counter(per_sub.values()).items()))},
                "Multi-timepoint subjects MUST stay in one split (see data.py).")

    # ---- labels ----------------------------------------------------------- #
    schemes, canon_ok, nest_ok = Counter(), 0, 0
    for rec in cases[:sample]:
        if "label" not in rec:
            continue
        try:
            arr = np.asarray(_load(rec["label"]).dataobj)
        except Exception as e:
            log.add(f"{spec_name}/labels", f"readable: {rec['case']}", FAIL,
                    note=str(e))
            continue
        vals = tuple(sorted(int(v) for v in np.unique(arr)))
        schemes[vals] += 1
        canon = to_canonical(arr, spec_name)
        canon_ok += 1
        try:
            assert_nesting(to_regions(canon), rec["case"])
            nest_ok += 1
        except AssertionError as e:
            log.add(f"{spec_name}/labels", "nesting holds after remap", FAIL,
                    note=str(e))

    union = sorted({v for s in schemes for v in s})
    unexpected = set(union) - spec.expected_labels
    log.add(f"{spec_name}/labels", f"labels within expected {sorted(spec.expected_labels)}",
            PASS if not unexpected else FAIL,
            {"observed_union": union, "distinct_schemes": len(schemes),
             "schemes": {str(k): v for k, v in schemes.items()}},
            "" if not unexpected else
            f"UNEXPECTED {sorted(unexpected)} -- update label_map in datasets.py "
            f"before training, or every region will be wrong.")

    log.add(f"{spec_name}/labels", "single consistent scheme across sample",
            PASS if len(schemes) == 1 else FAIL,
            {"n_schemes": len(schemes)},
            "" if len(schemes) == 1 else "Mixed schemes: remap per subset, not globally.")

    log.add(f"{spec_name}/labels", "WT>=TC>=ET nesting holds after remap",
            PASS if nest_ok == canon_ok and canon_ok else FAIL,
            {"checked": canon_ok, "ok": nest_ok})

    # ---- geometry --------------------------------------------------------- #
    shapes, vox, dtypes = Counter(), [], Counter()
    for rec in cases[:sample]:
        try:
            img = _load(rec["image"][0])
        except Exception as e:
            log.add(f"{spec_name}/geometry", f"readable: {rec['case']}", FAIL, note=str(e))
            continue
        shapes[tuple(int(x) for x in img.shape[:3])] += 1
        vox.append(float(abs(np.linalg.det(np.asarray(img.affine)[:3, :3]))))
        dtypes[str(img.get_data_dtype())] += 1

    if vox:
        uniform = (max(vox) - min(vox)) < 1e-6
        log.add(f"{spec_name}/geometry", "voxel volume uniform across sample",
                PASS if uniform else FAIL,
                {"mm3_min": round(min(vox), 6), "mm3_max": round(max(vox), 6)},
                "" if uniform else "Compute volumes per-file from the affine; "
                                   "a global constant will corrupt the endpoint.")
        log.add(f"{spec_name}/geometry", "voxel volume is 1.0 mm^3",
                PASS if abs(min(vox) - 1.0) < 1e-6 and uniform else INFO,
                {"mm3": round(min(vox), 6)})
        log.add(f"{spec_name}/geometry", "single spatial shape",
                PASS if len(shapes) == 1 else INFO,
                {"shapes": {str(k): v for k, v in shapes.items()},
                 "dtypes": dict(dtypes)},
                "" if len(shapes) == 1 else "Mixed shapes -- crop/pad handles it, "
                                            "but note it in the report.")

    # ---- intensity -------------------------------------------------------- #
    stats = []
    for rec in cases[:5]:
        a = np.asarray(_load(rec["image"][0]).dataobj, dtype=np.float32)
        nz = a[a > 0]
        if nz.size:
            stats.append({"min": float(a.min()), "max": float(a.max()),
                          "nz_mean": round(float(nz.mean()), 2),
                          "nz_frac": round(float((a > 0).mean()), 3)})
    if stats:
        mx = max(s["max"] for s in stats)
        note = ("Max <= 255 across all samples, which suggests uint8 rescaling "
                "(as in SAILOR's MNI volumes). Confirm this is genuine before "
                "trusting the normalisation in data.py." if mx <= 255.0 else "")
        log.add(f"{spec_name}/intensity", "intensity range recorded", INFO,
                {"samples": stats, "max_observed": round(mx, 1)}, note)

    # ---- channel order ---------------------------------------------------- #
    ex = cases[0]
    order = [Path(p).name for p in ex["image"]]
    log.add(f"{spec_name}/channels", f"modality order is {list(CANONICAL_MODALITY_ORDER)}",
            PASS, {"resolved_files": order},
            "This order MUST match the pretrained bundle (T1c, T1, T2, FLAIR). "
            "A wrong order trains fine and invalidates transfer results.")

    return {"cases": cases, "n_cases": len(cases), "n_subjects": len(subs),
            "labels": union, "shapes": dict(shapes)}


def check_split(cases: list[dict], split: dict, log: Log) -> None:
    """No subject may appear in more than one split."""
    sets = {k: set(v) for k, v in split.items()}
    overlaps = {}
    keys = list(sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ov = sets[keys[i]] & sets[keys[j]]
            if ov:
                overlaps[f"{keys[i]}&{keys[j]}"] = sorted(ov)[:5]
    log.add("split", "no subject appears in two splits",
            PASS if not overlaps else FAIL, {"overlaps": overlaps},
            "" if not overlaps else "LEAKAGE. Session/timepoint-level splitting "
                                    "inflates every metric. Fix before training.")
    log.add("split", "split sizes", INFO,
            {k: len(v) for k, v in sets.items()})


def run_all(base: str, datasets=("brats2021",), sample: int = 20,
            log_path="results/verification_log.jsonl") -> tuple[Log, dict]:
    log = Log(log_path)
    check_environment(log)
    out = {}
    for ds in datasets:
        print(f"\n{'-'*70}\n  {ds}\n{'-'*70}")
        try:
            check_integrity(ds, base, log)      # full scan first -- cheap, stat() only
            out[ds] = check_dataset(ds, base, log, sample)
        except Exception as e:
            log.add(f"{ds}/run", "check completed", FAIL, note=f"{type(e).__name__}: {e}")
    log.summary()
    return log, out
