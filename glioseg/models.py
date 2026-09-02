"""Five architecture arms behind one interface, plus the params/size/VRAM
accounting the rubric requires.

Every arm takes len(modalities) input channels and emits cfg.n_out channels of
RAW LOGITS. Sigmoid is applied in the loss and in evaluation, never inside the
model, so the same checkpoint works for both.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config

# Smallest patch each arm tolerates. SwinUNETR downsamples 32x, so below 64^3
# its bottleneck collapses to 1x1x1 and normalisation raises. Found by testing;
# not documented upstream.
MIN_PATCH = {"unet3d": 32, "segresnet": 32, "segformer3d": 32,
             "swinunetr": 64, "mednext": 32}

CITATIONS = {
    "unet3d":      "Cicek et al., MICCAI 2016, arXiv:1606.06650",
    "segresnet":   "Myronenko, BrainLes/MICCAI 2018, arXiv:1810.11654",
    "segformer3d": "Perera, Navard & Yilmaz, CVPR Workshops 2024, arXiv:2404.10156",
    "swinunetr":   "Hatamizadeh et al., 2022, arXiv:2201.01266",
    "mednext":     "Roy et al., MICCAI 2023, arXiv:2303.09975",
}


def build_model(cfg: Config) -> nn.Module:
    cin, cout = len(cfg.modalities), cfg.n_out
    kw = dict(cfg.model_kwargs)

    need = MIN_PATCH.get(cfg.model, 32)
    if min(cfg.patch_size) < need:
        raise ValueError(
            f"{cfg.model} requires patch >= {need}^3 (got {cfg.patch_size}); "
            f"its bottleneck collapses below that.")

    if cfg.model == "unet3d":
        from monai.networks.nets import UNet
        return UNet(spatial_dims=3, in_channels=cin, out_channels=cout,
                    channels=kw.pop("channels", (32, 64, 128, 256, 512)),
                    strides=kw.pop("strides", (2, 2, 2, 2)),
                    num_res_units=kw.pop("num_res_units", 2),
                    norm=kw.pop("norm", "INSTANCE"), **kw)

    if cfg.model == "segresnet":
        from monai.networks.nets import SegResNet
        return SegResNet(spatial_dims=3, in_channels=cin, out_channels=cout,
                         init_filters=kw.pop("init_filters", 16),
                         blocks_down=kw.pop("blocks_down", (1, 2, 2, 4)),
                         blocks_up=kw.pop("blocks_up", (1, 1, 1)),
                         dropout_prob=kw.pop("dropout_prob", 0.2), **kw)

    if cfg.model == "segformer3d":
        from .segformer3d import SegFormer3D
        return SegFormer3D(in_channels=cin, num_classes=cout, **kw)

    if cfg.model == "swinunetr":
        from monai.networks.nets import SwinUNETR
        common = dict(in_channels=cin, out_channels=cout,
                      feature_size=kw.pop("feature_size", 48),
                      use_checkpoint=kw.pop("use_checkpoint", True))
        # MONAI deprecated img_size in 1.3 and REMOVED it in 1.5, while earlier
        # versions require it. Try both so the code survives either pin.
        try:
            return SwinUNETR(**common, **kw)
        except TypeError:
            return SwinUNETR(img_size=cfg.patch_size, **common, **kw)

    if cfg.model == "mednext":
        # Vendored under glioseg/vendor/mednext/. The upstream repo is a fork of
        # the nnU-Net v1 pipeline and its standalone import path is documented as
        # broken (MedNeXt issue #22), so we import the architecture module only.
        # Use kernel 3 ONLY: k5 requires UpKern init from a trained k3 model,
        # which doubles training cost.
        try:
            from .vendor.mednext.MedNextV1 import MedNeXt
        except Exception as e:
            raise ImportError(
                "MedNeXt not vendored. Run scripts/vendor_mednext.sh, or drop "
                f"this arm -- it is optional. ({e})")
        return MedNeXt(in_channels=cin, n_classes=cout,
                       n_channels=kw.pop("n_channels", 32),
                       exp_r=kw.pop("exp_r", 2),
                       kernel_size=kw.pop("kernel_size", 3),   # k3 only
                       deep_supervision=False,
                       do_res=True, do_res_up_down=True,
                       block_counts=kw.pop("block_counts", [2]*9), **kw)

    raise ValueError(f"unknown model '{cfg.model}'")


# --------------------------------------------------------------------------- #
# Accounting for the rubric's "# parameters, model size, training time" table
# --------------------------------------------------------------------------- #

def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    nbytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return {"params_total": total, "params_trainable": train,
            "params_M": round(total / 1e6, 3),
            "model_size_MB": round(nbytes / 2**20, 2)}


def measure_vram(model: nn.Module, cfg: Config, train_pass: bool = True) -> dict:
    """Peak VRAM for one step. Report next to params -- it decides what fits."""
    if not torch.cuda.is_available():
        return {"peak_MB": None, "note": "no CUDA"}
    dev = "cuda"
    model = model.to(dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    x = torch.randn(cfg.batch_size, len(cfg.modalities), *cfg.patch_size, device=dev)
    try:
        if train_pass:
            model.train()
            with torch.autocast(dev, dtype=torch.float16, enabled=cfg.amp):
                out = model(x)
                loss = out.float().mean()
            loss.backward()
        else:
            model.eval()
            with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=cfg.amp):
                model(x)
        peak = torch.cuda.max_memory_allocated(dev) / 2**20
        res = {"peak_MB": round(peak, 1), "pass": "train" if train_pass else "eval"}
    except torch.cuda.OutOfMemoryError:
        res = {"peak_MB": None, "note": "OOM at this patch/batch"}
    finally:
        model.zero_grad(set_to_none=True)
        del x
        torch.cuda.empty_cache()
    return res


def summarize_arm(cfg: Config) -> dict:
    m = build_model(cfg)
    info = {"model": cfg.model, "citation": CITATIONS.get(cfg.model, "?"),
            **count_params(m)}
    info.update(measure_vram(m, cfg))
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return info


def shape_check(cfg: Config) -> dict:
    """Assert the arm round-trips shape. Run before committing GPU hours."""
    m = build_model(cfg).eval()
    x = torch.randn(1, len(cfg.modalities), *cfg.patch_size)
    with torch.no_grad():
        y = m(x)
    ok = tuple(y.shape) == (1, cfg.n_out, *cfg.patch_size)
    return {"model": cfg.model, "in": tuple(x.shape), "out": tuple(y.shape), "ok": ok}
