# 499A — 3D Glioma Segmentation

Codebase for the NSU CSE/EEE 499A senior design report. **Due 9 Sep 2026, 23:59.**

> **Benchmarking 3D architectures for glioma segmentation: efficiency, data scale,
> and transfer to post-treatment MRI.**

Three sub-questions, ordered by how certainly they get answered:

| | Question | Needs | Certainty |
|---|---|---|---|
| **Q1** | Does parameter count predict Dice? | BraTS only | high |
| **Q2** | How much training data does each arm need? | BraTS only | high |
| **Q3** | Does pre-op pretraining transfer to post-treatment? | + MU-Glioma-Post | the novelty |

Q1 + Q2 alone is a complete report. Q3 is the contribution.

---

## Quickstart (Colab)

Open `notebooks/00_verify.ipynb`, set Runtime → GPU, run top to bottom.
Then `01_train.ipynb`, then `02_analyse.ipynb`.

```bash
pip install -r requirements.txt
```

**Run 00_verify before spending a single compute unit.** It exists to catch the
one failure mode that costs the most: a label, affine, channel-order or intensity
inconsistency that trains without error and silently invalidates everything.

---

## Verified dataset facts

Checked against Drive on 2026-09-02, not assumed.

```
BraTS2021_Training_Data/BraTS2021_00000/
  BraTS2021_00000_{flair,t1,t1ce,t2,seg}.nii.gz      ~2.5 MB each
  Case IDs NON-CONTIGUOUS: 00000, 00002, 00003, 00005 ... 01467, 01665

MU-Glioma-Post/PatientID_0007/Timepoint_2/
  PatientID_0007_Timepoint_2_brain_{t1n,t1c,t2w,t2f}.nii.gz   ~5.4 MB each
  PatientID_0007_Timepoint_2_tumorMask.nii.gz
  Timepoints do NOT start at 1 for every patient and counts vary
```

**The two datasets use different naming conventions.** BraTS 2021 is
`flair/t1/t1ce/t2` + `_seg`; MU uses the 2023+ convention `t2f/t1n/t1c/t2w` +
`_tumorMask`. Any tutorial written for BraTS 2023 will silently find zero cases
in your BraTS 2021 folder.

---

## Measured on this machine

By running, not by reading papers:

| Arm | Params | Size | Min patch | Grad-CAM target | Citation |
|---|---|---|---|---|---|
| SegResNet | 4.70 M | 17.9 MB | 32³ | `down_layers.3` | Myronenko, BrainLes 2018 |
| SegFormer3D | 4.51 M | 17.2 MB | 32³ | `fuse` | Perera et al., CVPR-W 2024 |
| 3D U-Net | 19.22 M | 73.3 MB | 32³ | deep encoder block | Çiçek et al., MICCAI 2016 |
| SwinUNETR | 62.19 M | 244.4 MB | **64³** | `encoder10` | Hatamizadeh et al., 2022 |
| MedNeXt-S k3 | ~5.6 M | — | 32³ | `dec_block_0` | Roy et al., MICCAI 2023 |

SegFormer3D's 4.51 M matches the paper's reported 4.5 M.

---

## Non-obvious things encoded here

Each of these cost real debugging time and is now a guard rail.

**Sigmoid, not softmax.** ET ⊂ TC ⊂ WT are *nested*: one voxel belongs to all
three. A softmax head forces one label per voxel, contradicting the endpoint
definition and making outputs incomparable to every published ET/TC/WT number.
`regions.py` emits 3 independent sigmoid channels.

**Cross-version label alignment is free once you use regions.** BraTS 2021 encodes
ET as 4, MU as 3. Regions are unions, so after remapping to canonical integers the
numbering stops mattering. Verified: identical anatomy in both native schemes
produces byte-identical WT/TC/ET.

**Per-channel Dice, never flattened.** Flattening collapses three regions into one
scalar so small ET is drowned by large WT. Tested: an ET-only failure raises the
loss from 0.00 to 0.35.

**SwinUNETR needs patch ≥ 64³.** Below that its bottleneck collapses to 1×1×1 and
normalisation raises. Encoded in `models.MIN_PATCH`; not documented upstream.

**Grad-CAM must target conv layers.** Transformer blocks emit `(B,N,C)` tokens with
no spatial grid to pool over. `xai.pick_target_layer` handles this and raises with
a clear message if you repoint it wrongly.

**MONAI is pinned to 1.4.x.** `SwinUNETR`'s `img_size` argument was deprecated in
1.3 and *removed* in 1.5, while earlier versions require it. `build_model` tries
both signatures anyway.

**fp16, not bf16.** T4 is Turing and has no bf16. Loss is computed in fp32
(`logits.float()`) because Dice/BCE NaN under fp16, and Dice keeps a smooth term
in both numerator and denominator so an all-background patch doesn't divide by ~0.

**Checkpoint every epoch, atomic rename.** Colab disconnects around 12 hours and
you are paying for the GPU. Temp-file-plus-rename means a disconnect mid-write
cannot corrupt the checkpoint.

**Tar once, copy-and-untar per session.** 6,250 small files over mounted Drive is
the pathological I/O case. One large sequential file is minutes.

**Subject-level splits only.** MU has multiple timepoints per patient; splitting at
timepoint level leaks and inflates every metric. `data.make_split` groups by
subject and `verify.check_split` asserts no overlap.

**Modality order is T1c, T1, T2, FLAIR.** This matches the MONAI pretrained
bundles. A wrong order trains fine and invalidates the transfer experiment.

---

## Layout

```
glioseg/
  datasets.py     registry with VERIFIED naming conventions per dataset
  verify.py       Tier-0 checks; run before anything else
  regions.py      canonical remap + nested WT/TC/ET channels
  config.py       Config + the experiment grid per tier
  data.py         transforms, subject-level splits, PersistentDataset
  models.py       5-arm factory + params/size/VRAM accounting
  segformer3d.py  SegFormer3D (arXiv:2404.10156), validated at 4.51 M
  losses.py       per-channel Dice + BCEWithLogits, DiceFocal variant
  train.py        fp16 + GradScaler, checkpoint/resume, grid runner
  transfer.py     partial state_dict loading with matched/skipped reporting
  evaluate.py     sliding-window inference, per-channel scoring
  metrics.py      rubric metrics, volume stratification, Wilcoxon/Cohen's d/FDR
  postproc.py     connected-component filtering + nesting repair
  xai.py          Grad-CAM (conv targets) + occlusion + report figures
  tables.py       IEEE LaTeX emitters
notebooks/
  00_verify.ipynb   Tier 0 — no GPU
  01_train.ipynb    Tiers 1–5, 7
  02_analyse.ipynb  statistics, XAI, tables — no GPU
scripts/
  vendor_mednext.sh  optional 5th arm
```

---

## Rubric mapping

| Requirement | Produced by |
|---|---|
| Accuracy, precision, recall, F1, IoU, mAP | `tables.performance_table` |
| # params, model size, training time/epoch | `tables.efficiency_table` |
| XAI (Grad-CAM) | `02_analyse.ipynb` §2.4 |
| Ablation study | `tables.ablation_table` |
| Comparison with existing works | `tables.comparison_table` |
| Where trained, epochs, elapsed | every `result.json` |

Two things to write in the report. **Dice equals F1** for binary overlap — both
are emitted since the rubric names F1, so state the identity once. And
**Grad-CAM needed adapting**: it was defined for classifiers, so for segmentation
we differentiate the summed channel logit over the predicted region, following
Vinogradova et al., AAAI 2020.

---

## Status

✅ Validated by running: dataset discovery on both layouts, `_t1ce`/`_t1`
disambiguation, label remap and nesting, per-channel losses, all four arms
shape-checked and Grad-CAM'd, connected-component nesting repair, LaTeX emitters,
and a full train → evaluate → tables → resume cycle.

⚠️ **Never run against real BraTS or MU data.** Naming conventions were verified
via the Drive API but no volume has been opened. Expect Tier 0 to surface
something on first contact — that is what it is for.

⚠️ The external numbers in `tables.comparison_table` are **placeholders** and are
validation-set, not hidden-test. Verify each against its source before submitting.

⚠️ `mAP` for segmentation is our reading of the rubric (average precision over a
swept probability threshold), not a standard BraTS metric. Define it explicitly.
