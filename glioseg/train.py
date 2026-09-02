"""Training with fp16 AMP and crash-safe checkpointing.

Colab Pro disconnects at roughly 12 hours and you are paying for the GPU, so a
lost run costs money. Every epoch writes a resumable checkpoint via a temp file
plus atomic rename, so a disconnect mid-write cannot corrupt it.

T4 is Turing and has NO bf16. This module uses fp16 + GradScaler and computes
the loss in fp32; both are required for stable Dice/BCE under autocast.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import build_loaders
from .evaluate import evaluate
from .losses import build_loss
from .models import build_model, count_params, measure_vram
from .transfer import load_pretrained

CKPT_LAST, CKPT_BEST = "last.pt", "best.pt"


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _atomic_save(obj, path: Path):
    """Write to a temp file then rename. Rename is atomic on the same
    filesystem, so a disconnect cannot leave a half-written checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_ckpt(cfg: Config, model, opt, sched, scaler, epoch, best, history, tag=CKPT_LAST):
    _atomic_save({
        "epoch": epoch, "best": best, "history": history,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched else None,
        "scaler": scaler.state_dict() if scaler else None,
        # RNG states must be saved as CPU ByteTensors. torch.save/load can
        # otherwise hand back a type set_rng_state rejects ("RNG state must be
        # a torch.ByteTensor"), which silently costs exact reproducibility --
        # and that matters for the multi-seed statistics in Tier 4.
        "rng": {"torch": torch.get_rng_state().cpu().to(torch.uint8),
                "cuda": [t.cpu().to(torch.uint8) for t in torch.cuda.get_rng_state_all()]
                        if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(), "python": random.getstate()},
        "config_hash": cfg.hash(),
    }, cfg.run_dir / tag)


def maybe_resume(cfg: Config, model, opt, sched, scaler):
    """Returns (start_epoch, best, history). 0/-1/[] if nothing to resume."""
    p = cfg.run_dir / CKPT_LAST
    if not p.exists():
        return 0, -1.0, []
    ck = torch.load(p, map_location=_device(), weights_only=False)
    if ck.get("config_hash") != cfg.hash():
        print(f"  [resume] hash mismatch -- ignoring stale checkpoint at {p}")
        return 0, -1.0, []
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    if sched and ck.get("sched"):
        sched.load_state_dict(ck["sched"])
    if scaler and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    r = ck.get("rng", {})
    try:
        torch.set_rng_state(torch.as_tensor(r["torch"], dtype=torch.uint8).cpu())
        if r.get("cuda") and torch.cuda.is_available():
            states = [torch.as_tensor(t, dtype=torch.uint8).cpu() for t in r["cuda"]]
            if len(states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(states)
        np.random.set_state(r["numpy"])
        random.setstate(r["python"])
    except Exception as e:
        print(f"  [resume] RNG restore skipped ({e}) -- training continues, but "
              f"this run is not bit-exact reproducible from the checkpoint")
    print(f"  [resume] continuing from epoch {ck['epoch']+1}, best={ck['best']:.4f}")
    return ck["epoch"] + 1, ck["best"], ck.get("history", [])


def train_arm(cfg: Config, split_path=None, verbose=True) -> dict:
    dev = _device()
    set_seed(cfg.seed)
    cfg.save()

    tr, va, te, meta = build_loaders(cfg, split_path, verbose=verbose)
    model = build_model(cfg).to(dev)
    sizing = count_params(model)

    if cfg.pretrained:
        info = load_pretrained(model, cfg.pretrained, verbose=verbose)
        sizing["pretrained_match_frac"] = info["match_frac"]

    loss_fn = build_loss(cfg.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    # fp16 GradScaler. Enabled only on CUDA; a no-op scaler on CPU.
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and dev == "cuda")

    start, best, history = maybe_resume(cfg, model, opt, sched, scaler)

    if verbose:
        print(f"\n=== {cfg.name} | {cfg.model} | {sizing['params_M']}M params "
              f"| {sizing['model_size_MB']}MB | {dev} | patch {cfg.patch_size}")

    t0_all = time.time()
    for ep in range(start, cfg.epochs):
        model.train()
        t0, run, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)

        for i, batch in enumerate(tr):
            x = batch["image"].to(dev, non_blocking=True)
            y = batch["label"].to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.float16,
                                enabled=cfg.amp and dev == "cuda"):
                out = model(x)
            loss = loss_fn(out, y) / cfg.grad_accum      # loss in fp32 inside
            scaler.scale(loss).backward()
            if (i + 1) % cfg.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            run += loss.item() * cfg.grad_accum
            nb += 1

        sched.step()
        dt = time.time() - t0
        row = {"epoch": ep + 1, "train_loss": round(run / max(nb, 1), 5),
               "epoch_time_s": round(dt, 2), "lr": sched.get_last_lr()[0]}

        if (ep + 1) % cfg.val_every == 0 or ep + 1 == cfg.epochs:
            agg, _ = evaluate(model, va, cfg, with_ap=False)
            row["val_mean_dice"] = agg.get("mean_dice_mean")
            row["val_ET_dice"] = agg.get("ET_dice_mean")
            if (row["val_mean_dice"] or -1) > best:
                best = row["val_mean_dice"]
                _atomic_save({"model": model.state_dict(), "epoch": ep,
                              "best": best, "config_hash": cfg.hash()},
                             cfg.run_dir / CKPT_BEST)

        history.append(row)
        save_ckpt(cfg, model, opt, sched, scaler, ep, best, history)
        (cfg.run_dir / "history.json").write_text(json.dumps(history, indent=2))

        if verbose:
            msg = f"  ep {ep+1:>3}/{cfg.epochs}  loss {row['train_loss']:.4f}  {dt:.1f}s"
            if row.get("val_mean_dice") is not None:
                msg += f"  val_dice {row['val_mean_dice']:.4f}"
            print(msg, flush=True)

    total = time.time() - t0_all
    times = [r["epoch_time_s"] for r in history]

    # final evaluation on TEST with the best checkpoint
    bp = cfg.run_dir / CKPT_BEST
    if bp.exists():
        model.load_state_dict(torch.load(bp, map_location=dev,
                                         weights_only=False)["model"])
    agg, per_case = evaluate(model, te, cfg, with_ap=True)

    result = {
        "name": cfg.name, "model": cfg.model, "dataset": cfg.dataset,
        "config_hash": cfg.hash(), "seed": cfg.seed,
        "pretrained": cfg.pretrained, "loss": cfg.loss,
        "patch": list(cfg.patch_size), "n_modalities": len(cfg.modalities),
        "n_cases": meta["n_cases"], "split_counts": meta["counts"],
        **sizing,
        "epochs_run": len(history),
        "total_train_time_s": round(total, 1),
        "mean_epoch_time_s": round(float(np.mean(times)), 2) if times else None,
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        **agg,
    }
    (cfg.run_dir / "per_case.json").write_text(json.dumps(per_case, indent=2, default=str))
    (cfg.run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

    if verbose:
        print(f"  DONE  test mean Dice {agg.get('mean_dice_mean')}  "
              f"ET {agg.get('ET_dice_mean')}  {total/60:.1f} min -> {cfg.run_dir.name}")
    return result


def run_grid(configs: list[Config], split_path=None) -> list[dict]:
    """Run arms in sequence, surviving individual failures.

    One OOM must not kill an overnight grid you paid for.
    """
    out = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'='*70}\n[{i}/{len(configs)}] {cfg.name}  ({cfg.model})\n{'='*70}")
        try:
            out.append(train_arm(cfg, split_path))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            out.append({"name": cfg.name, "model": cfg.model,
                        "error": f"{type(e).__name__}: {e}"})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out
