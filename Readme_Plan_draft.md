# PLAN v5 — 499A, Verified & Executable

**Due:** 9 Sep 2026, 23:59 · **Compute:** Colab Pro only (T4 workhorse, A100 scalpel) · **Solo execution**
**Supersedes:** v4 and the action plan · **Dataset facts below are verified against your Drive, not assumed**

---

## 1. Verified dataset facts

| Fact | Value | How verified |
|---|---|---|
| Path | `MyDrive/Colab Notebooks/499a/BraTS2021_Training_Data/` | Drive listing |
| Layout | one folder per case, 5 files inside | Drive listing |
| Modality suffixes | `_flair`, `_t1`, `_t1ce`, `_t2` | **2021 convention — not 2023's `t2f/t1n/t1c/t2w`** |
| Seg suffix | `_seg.nii.gz` | Drive listing |
| Per-case size | ~10 MB (seg only 58 KB) | Drive listing |
| Case IDs | **non-contiguous** — gaps throughout | 00000, 00002, 00003, 00005… 01467, 01665 |
| Est. total | ~12.5 GB / ~6,250 files | 1251 × 5 × ~2.5 MB |

❓ **Still unverified, and Tier 0 must check:** label integers (expect `{0,1,2,4}`), volume shape (expect 240×240×155), voxel size (expect 1 mm³), and whether every case actually has all five files.

---

## 2. Dataset decision — two phases, one codebase

**BraTS 2021 is primary. MU-Glioma-Post hooks in at Phase 2.** Start on one dataset, but the loader is a registry with per-dataset label remapping from day one, so adding MU is a config entry, not a rewrite.

**Why MU still matters:** BraTS 2021 is entirely pre-operative. The pre-op → post-treatment transfer question — the actual novelty — needs a post-treatment target. Without MU you have an architecture comparison, which is a reproduction study. With it you have a domain-gap finding. MU is 11 GB and costs one extra fine-tune per arm.

**SAILOR** stays as optional external validation for the volume-stratified analysis. Lowest priority; it's your individual-contribution depth if time allows.

---

## 3. The research question, staged so it survives either way

> **Benchmarking 3D architectures for glioma segmentation: efficiency, data scale, and transfer to post-treatment MRI.**

Three sub-questions, in order of how certainly you'll answer them:

**Q1 — Architecture and efficiency (BraTS only, certain).** Does parameter count predict Dice? Four arms spanning 4.5M–62M, identical protocol, with params / model size / s-per-epoch measured.

**Q2 — Data efficiency (BraTS only, certain, and it *justifies your subsetting*).** Train on 150 / 300 / 600 / all cases. How much data does each architecture need before it saturates?

This is the save that turns a compute constraint into a finding. Nobody can object that you didn't use all 1,251 cases, because **the training-set size is the experiment.** It's also a rubric-perfect ablation.

**Q3 — Cross-domain transfer (needs MU, the novelty).** Do BraTS-pretrained weights transfer to post-treatment glioma, and does fine-tuning close the gap?

If Q3 doesn't happen, drop the third clause of the title. Q1 + Q2 is already a complete report.

---

## 4. Arms — MedNeXt last, as you asked

| Order | Arm | Citation | Params | Pretrained BraTS weights | GPU |
|---|---|---|---|---|---|
| 1 | **SegResNet** | Myronenko, BrainLes 2018 | ~4.7 M | ✅ MONAI `brats_mri_segmentation` (BraTS 2018, 4-in/3-out sigmoid) | T4 |
| 2 | **3D U-Net** | Çiçek et al., MICCAI 2016 | ~19 M | ❌ | T4 |
| 3 | **SegFormer3D** | Perera et al., **CVPR-W 2024** | 4.52 M ✅ measured | ❌ (train from scratch) | T4 |
| 4 | **SwinUNETR** | Hatamizadeh et al., 2022 | 62.2 M ✅ measured | ✅ MONAI research-contributions BRATS21 | **A100** |
| 5 *(last, optional)* | **MedNeXt-S k3** | Roy et al., **MICCAI 2023** | ~5.6 M | ❌ | T4 |

**Order matters:** SegResNet first because it has pretrained weights *and* is cheapest — it validates the whole pipeline at minimum cost. MedNeXt last because it's a fork of the nnU-Net v1 pipeline with a documented broken standalone import path, and **k5 requires UpKern init from a trained k3 model**, doubling cost. Use k3 only, and only if ahead.

⚠️ **Two hard constraints found in research:** SwinUNETR's `img_size` argument was deprecated in MONAI 1.3 and **removed in 1.5** — passing it raises, omitting it raises on older versions. Pin MONAI 1.4.x. And the pretrained bundles expect modality order **T1c, T1, T2, FLAIR** — if your loader stacks in a different order, the transfer experiment is meaningless while appearing to work.

---

## 5. Tiers with concrete actions

### TIER 0 — Verification · Day 1 · no GPU, ~2h
1. Mount Drive, enumerate case folders by **discovery** (glob, never a numeric range).
2. Assert all 5 files present per case; report and exclude incomplete cases.
3. `np.unique()` on 20 segmentations → confirm `{0,1,2,4}`.
4. Affine determinant → confirm 1 mm³. Shape → confirm 240×240×155.
5. **One-time tar:** archive the whole dataset to a single `.tar` on Drive.
6. Write `split_manifest.json` — patient-level 70/15/15, seed 42.

**Gate:** every assertion passes, or the remap table changes before any training.
**Output:** `dataset_report.json`, `brats2021.tar` on Drive, `split_manifest.json`

### TIER 1 — Pipeline validation · Day 2 · ~3 T4-h
1. Copy tar → local disk, untar, verify count.
2. Build 3-channel sigmoid head (WT/TC/ET), per-channel Dice+BCE loss.
3. Smoke run: SegResNet, 2 epochs, 300-case subset, 64³, batch 2.
4. **Verify checkpoint/resume by killing the session deliberately and resuming.**
5. Load the MONAI SegResNet bundle; log matched/skipped keys.

**Gate:** loss decreases, Dice non-zero, resume works, bundle loads.

### TIER 2 — Q1 architecture comparison · Days 3–6 · ~20 T4-h + 4 A100-h
Four arms, 300 cases, 64³, 30 epochs, identical protocol. Record params, model size, s/epoch, peak VRAM.
**Gate:** ≥3 arms complete. Failed arms are recorded, not hidden.

### TIER 3 — Q2 data efficiency · Days 5–7 · ~12 T4-h
Best arm at 150 / 300 / 600 / 1251 cases. **Justifies every subset used elsewhere.**
**Gate:** a curve with 4 points.

### TIER 4 — Statistics · Days 6–8 · ~6 T4-h
3 seeds on the primary comparison. Paired Wilcoxon, Cohen's d, patient-level bootstrap CIs.
**Gate:** you can say whether differences exceed seed noise.

### TIER 5 — Ablations · Days 7–9 · ~8 T4-h
Loss (DiceCE vs DiceFocal) · modalities (4 vs T1c+FLAIR) · **post-processing on/off** (inference-only, near-free) · patch 64³ vs 96³.

### TIER 6 — XAI · Day 8 · ~1 h · **mandatory**
Grad-CAM on the best arm. Target **conv layers only** — transformer blocks emit `(B,N,C)` tokens with no spatial grid. Occlusion sensitivity as cross-check.

### TIER 7 — Q3 transfer, the novelty · Days 8–10 · ~6 T4-h
Only if Tiers 0–5 are done. Download MU-Glioma-Post (11 GB), register its label map, run A0 (scratch) vs A2 (BraTS-init fine-tune) for SegResNet + one more arm.

### TIER 8 — Writing · Days 9–11
Start day 9 at the latest.

---

## 6. Codebase modules

```
glioseg/
  config.py       Config dataclass, hashing, experiment grids
  datasets.py     REGISTRY: per-dataset paths, modality aliases, label maps
  verify.py       Tier-0 checks — runs before anything else
  regions.py      canonical remap + nested WT/TC/ET channels
  data.py         MONAI transforms, patient-level splits, PersistentDataset
  models.py       5-arm factory + params/size/VRAM accounting
  segformer3d.py  already written and validated (4.52M)
  losses.py       per-channel Dice + BCEWithLogits, DiceFocal variant
  train.py        fp16 + GradScaler, checkpoint/resume to Drive
  transfer.py     partial state_dict loading with matched/skipped reporting
  evaluate.py     sliding-window inference, per-channel metrics
  metrics.py      rubric metrics + volume stratification + Wilcoxon/Cohen's d
  postproc.py     connected-component filtering (Tier 5 ablation)
  xai.py          Grad-CAM (conv targets) + occlusion + figures
  tables.py       LaTeX emitters
notebooks/
  00_verify.ipynb   Tier 0
  01_train.ipynb    Tiers 1–5, one arm per run
  02_analyse.ipynb  Tiers 6–8, tables and figures
```

**Delivery:** GitHub. Your workflow is human-in-the-loop — you run, paste output, I patch. With git that's `git pull`; with Drive it's a re-upload every iteration.

---

## 7. Engineering decisions locked in

| Decision | Why |
|---|---|
| MONAI pinned to 1.4.x | SwinUNETR `img_size` removed in 1.5 |
| fp16 + `GradScaler`, **not bf16** | T4 is Turing; bf16 unsupported |
| Loss computed in fp32 (`logits.float()`) | Dice/BCE NaN under fp16 |
| `BCEWithLogits`, never sigmoid-then-BCE | numerical stability under autocast |
| Dice `smooth_nr` and `smooth_dr` both > 0 | all-background patches divide by ~0 |
| Checkpoint **every epoch**, atomic rename | Colab disconnects ~12 h; a lost paid run is the worst outcome |
| Tar once, copy-and-untar per session | 6,250 small files over Drive is pathological |
| `PersistentDataset` on local disk | 12.5 GB won't fit comfortably in RAM cache |
| Modality order **T1c, T1, T2, FLAIR** | matches pretrained bundles; wrong order silently invalidates transfer |

---

## 8. What I need from you

1. **GitHub repo** — create an empty one and give me the URL, or say you'd rather use Drive.
2. **MU-Glioma-Post** — downloaded already, or a Tier-7 TODO?
3. **Confirm the parent folder** for outputs — I assume `MyDrive/Colab Notebooks/499a/`.

None of these block me writing the codebase; they only affect the notebook's paths.

---

## 9. Biggest risk

**A silent modality-order or label-remap mismatch that trains without error and invalidates everything.** BraTS uses ET=4, MU uses ET=3 with RC=4, and the pretrained bundles expect a specific channel order. Tier 0 exists solely to catch this, and every dataset load asserts it.
