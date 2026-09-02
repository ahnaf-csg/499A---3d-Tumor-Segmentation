"""Experiment configuration. One Config fully determines a run and is hashed
into the artifact directory name, so provenance is a chain you can walk back.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Literal

from .datasets import CANONICAL_MODALITY_ORDER

ModelName = Literal["unet3d", "segresnet", "segformer3d", "swinunetr", "mednext"]
LossName = Literal["dice_bce", "dice_focal", "dice"]

# Drive layout, verified 2026-09-02
DRIVE_BASE = "/content/drive/MyDrive/Colab Notebooks/499a"
LOCAL_BASE = "/content/data"


@dataclass
class Config:
    name: str = "run"
    seed: int = 42

    # --- data ---
    dataset: str = "brats2021"
    data_base: str = LOCAL_BASE          # where the dataset ROOTS live
    modalities: tuple[str, ...] = CANONICAL_MODALITY_ORDER
    include_rc: bool = False             # 4th channel; only meaningful for mu_post
    n_cases: int | None = None           # None = all. Tier-3 data-efficiency knob.
    patch_size: tuple[int, int, int] = (64, 64, 64)
    cache_dir: str | None = "/content/cache"   # PersistentDataset; None disables
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)

    # --- model ---
    model: ModelName = "segresnet"
    model_kwargs: dict = field(default_factory=dict)
    pretrained: str | None = None        # path to a checkpoint for fine-tuning

    # --- optimisation ---
    loss: LossName = "dice_bce"
    epochs: int = 30
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-5
    amp: bool = True                     # fp16 + GradScaler (T4 has no bf16)
    grad_accum: int = 1
    samples_per_volume: int = 2          # RandCropByPosNegLabeld num_samples

    # --- eval ---
    num_workers: int = 4                 # data loading is often the bottleneck
    sw_batch_size: int = 2
    sw_overlap: float = 0.5
    val_every: int = 5
    postproc_min_voxels: int = 0         # 0 = off; Tier-5 ablation turns it on

    # --- output ---
    out_root: str = f"{DRIVE_BASE}/artifacts"

    @property
    def n_out(self) -> int:
        return 4 if self.include_rc else 3

    def hash(self) -> str:
        d = asdict(self)
        d.pop("out_root", None)          # moving output dir must not change identity
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]

    # Fields that actually change the CACHED tensors. The deterministic transform
    # prefix depends only on these -- not on model, loss, lr or epochs.
    _DATA_FIELDS = ("dataset", "modalities", "include_rc", "n_cases",
                    "patch_size", "split", "seed")

    def data_hash(self) -> str:
        """Cache key for PersistentDataset.

        Keying the cache on the FULL config would give every architecture its own
        copy. At ~60 MB per cached case that is ~55 GB per arm on the full BraTS
        set, so a 3-arm grid would need ~165 GB and run the Colab disk dry
        mid-grid. Keying on data-relevant fields only means arms 2 and 3 reuse
        arm 1's cache -- one copy, and they start fast.
        """
        d = {k: getattr(self, k) for k in self._DATA_FIELDS}
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]

    @property
    def run_dir(self) -> Path:
        return Path(self.out_root) / f"{self.name}_{self.model}_{self.hash()}"

    def save(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        p = self.run_dir / "config.json"
        p.write_text(json.dumps(asdict(self), indent=2, default=str))
        return p

    def variant(self, **kw) -> "Config":
        return replace(self, **kw)


# --------------------------------------------------------------------------- #
# Experiment grids, one per report table.
# --------------------------------------------------------------------------- #

def tier2_architectures(base: Config, arms=("segresnet", "unet3d", "segformer3d",
                                            "swinunetr")) -> list[Config]:
    """Q1: architecture comparison. Only `model` varies, so any difference is
    attributable to the architecture. SwinUNETR needs patch >= 64."""
    return [base.variant(model=m, name=f"{base.name}-arch") for m in arms]


def tier3_data_efficiency(base: Config, sizes=(150, 300, 600, None)) -> list[Config]:
    """Q2: how much data does the best arm need? Also retroactively justifies
    every subset used elsewhere -- the training-set size IS the experiment."""
    return [base.variant(n_cases=n, name=f"{base.name}-n{n or 'all'}") for n in sizes]


def tier4_seeds(base: Config, seeds=(42, 1, 2)) -> list[Config]:
    """Statistical rigour on the primary comparison only."""
    return [base.variant(seed=s, name=f"{base.name}-s{s}") for s in seeds]


def tier5_ablations(base: Config) -> list[Config]:
    """One factor at a time. Not a cartesian product -- won't fit the schedule
    and the rubric wants a readable table."""
    out = []
    for l in ("dice", "dice_bce", "dice_focal"):
        out.append(base.variant(loss=l, name=f"{base.name}-loss_{l}"))
    for mods in (("t1c", "t2f"), CANONICAL_MODALITY_ORDER):
        out.append(base.variant(modalities=mods, name=f"{base.name}-mod{len(mods)}"))
    for ps in ((64, 64, 64), (96, 96, 96)):
        out.append(base.variant(patch_size=ps, name=f"{base.name}-patch{ps[0]}"))
    return out


def tier7_transfer(base: Config, pretrained_path: str,
                   arms=("segresnet",)) -> list[Config]:
    """Q3, the novelty: pre-op -> post-treatment transfer.
    A0 = random init on target; A2 = pre-op weights fine-tuned on target."""
    out = []
    for m in arms:
        out.append(base.variant(model=m, dataset="mu_post", pretrained=None,
                                name=f"{base.name}-{m}-A0"))
        out.append(base.variant(model=m, dataset="mu_post", pretrained=pretrained_path,
                                lr=1e-5, name=f"{base.name}-{m}-A2"))
    return out
