"""Dataset registry.

Every convention here was VERIFIED against the actual Drive contents on
2026-09-02, not inferred from documentation. If a dataset is added, add a
DatasetSpec and run `verify.run_all` before training on it.

Verified layouts
----------------
BraTS 2021   BraTS2021_Training_Data/BraTS2021_00000/
               BraTS2021_00000_{flair,t1,t1ce,t2,seg}.nii.gz     (~2.5 MB each)
             Case IDs are NON-CONTIGUOUS (00000, 00002, 00003, 00005, ... 01665)

MU-Glioma    MU-Glioma-Post/PatientID_0007/Timepoint_2/
-Post          PatientID_0007_Timepoint_2_brain_{t1n,t1c,t2w,t2f}.nii.gz
               PatientID_0007_Timepoint_2_tumorMask.nii.gz       (~5.4 MB each)
             Timepoints do NOT start at 1 for every patient (0007 starts at 2)
             and counts differ per patient. Enumerate by discovery only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------- #
# Canonical label space. Everything is remapped into this before use.
# --------------------------------------------------------------------------- #
CANONICAL = {0: "background", 1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}

# Nested evaluation regions. ET subset TC subset WT, so channels OVERLAP and the
# head must be sigmoid, never softmax.
REGIONS: dict[str, tuple[int, ...]] = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
}
REGION_NAMES = list(REGIONS)

# Modality order expected by the MONAI pretrained bundles. Getting this wrong
# silently invalidates every transfer result while training without error.
CANONICAL_MODALITY_ORDER = ("t1c", "t1n", "t2w", "t2f")


@dataclass
class DatasetSpec:
    name: str
    root: str
    # canonical modality key -> filename fragment for THIS dataset
    modality_suffix: dict[str, str]
    seg_suffix: str
    # native integer -> canonical integer
    label_map: dict[int, int]
    # how deep case directories sit below root (1 = root/case, 2 = root/pt/tp)
    case_depth: int
    subject_regex: str
    expected_labels: set[int] = field(default_factory=set)
    notes: str = ""

    def subject_of(self, case_id: str) -> str:
        m = re.search(self.subject_regex, case_id)
        if not m:
            raise ValueError(f"[{self.name}] cannot extract subject from '{case_id}'")
        return m.group(0)


REGISTRY: dict[str, DatasetSpec] = {
    "brats2021": DatasetSpec(
        name="brats2021",
        root="BraTS2021_Training_Data",
        modality_suffix={"t1c": "_t1ce", "t1n": "_t1", "t2w": "_t2", "t2f": "_flair"},
        seg_suffix="_seg",
        # BraTS pre-2023: 1=NCR/NET, 2=ED, 4=ET, no label 3. ET moves 4 -> 3.
        label_map={0: 0, 1: 1, 2: 2, 4: 3},
        case_depth=1,
        subject_regex=r"BraTS2021_\d+",
        expected_labels={0, 1, 2, 4},
        notes="pre-operative; SOURCE domain. '_t1' must be matched AFTER '_t1ce'.",
    ),
    "mu_post": DatasetSpec(
        name="mu_post",
        root="MU-Glioma-Post",
        modality_suffix={"t1c": "_brain_t1c", "t1n": "_brain_t1n",
                         "t2w": "_brain_t2w", "t2f": "_brain_t2f"},
        seg_suffix="_tumorMask",
        # BraTS-2024 scheme, already canonical. Written out so the assumption
        # is visible in code rather than implied.
        label_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        case_depth=2,
        subject_regex=r"PatientID_\d+",
        expected_labels={0, 1, 2, 3, 4},
        notes="post-treatment; TARGET domain. RC=4 exists here only. "
              "Timepoints are non-contiguous and vary per patient.",
    ),
}


def resolve_root(spec: DatasetSpec, base: str | Path) -> Path:
    return Path(base) / spec.root


def find_cases(spec: DatasetSpec, base: str | Path,
               modalities: Sequence[str] = CANONICAL_MODALITY_ORDER,
               require_seg: bool = True, verbose: bool = True) -> list[dict]:
    """Discover cases. Never assumes ID ranges or contiguous numbering.

    Returns [{"case", "subject", "image": [paths in `modalities` order], "label"}].
    Images are returned in the order given by `modalities`, which defaults to the
    order the pretrained bundles expect.
    """
    root = resolve_root(spec, base)
    if not root.exists():
        raise FileNotFoundError(f"[{spec.name}] root not found: {root}")

    # a case directory is one that directly contains NIfTI files
    case_dirs = sorted({p.parent for p in root.rglob("*.nii*")})
    cases, skipped = [], {}

    for d in case_dirs:
        files = sorted(p for p in d.iterdir() if p.name.endswith((".nii", ".nii.gz")))
        names = {p.name: p for p in files}

        imgs, missing = [], []
        for m in modalities:
            frag = spec.modality_suffix[m]
            # longest-fragment-first so '_t1ce' wins over '_t1'
            hit = None
            for cand in sorted(names, key=len, reverse=True):
                if frag + "." in cand:
                    hit = names[cand]
                    break
            if hit is None:
                missing.append(m)
            else:
                imgs.append(str(hit))
        if missing:
            skipped[d.name] = f"missing {missing}"
            continue

        seg = next((str(v) for k, v in names.items() if spec.seg_suffix + "." in k), None)
        if require_seg and seg is None:
            skipped[d.name] = "missing seg"
            continue

        case_id = d.name if spec.case_depth == 1 else f"{d.parent.name}__{d.name}"
        rec = {"case": case_id, "subject": spec.subject_of(case_id), "image": imgs}
        if seg:
            rec["label"] = seg
        cases.append(rec)

    if verbose:
        print(f"[find_cases:{spec.name}] {root}")
        print(f"  usable cases : {len(cases)}")
        print(f"  subjects     : {len({c['subject'] for c in cases})}")
        if skipped:
            print(f"  SKIPPED {len(skipped)}: {dict(list(skipped.items())[:5])}"
                  f"{' ...' if len(skipped) > 5 else ''}")
        if cases:
            ex = cases[0]
            print(f"  example      : {ex['case']}  (subject {ex['subject']})")
            for m, p in zip(modalities, ex["image"]):
                print(f"      {m:4s} <- {Path(p).name}")
            print(f"      seg  <- {Path(ex.get('label','-')).name}")

    if not cases:
        raise RuntimeError(
            f"[{spec.name}] no usable cases under {root}. Inspect the tree and fix "
            f"modality_suffix / seg_suffix in datasets.py -- do not guess."
        )
    return cases
