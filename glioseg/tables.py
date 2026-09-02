r"""IEEE LaTeX table emitters, one per rubric-named table.

Every table is \label'd so \ref{} works. External comparison numbers are
PLACEHOLDERS -- verify each against its source before submitting, and note
whether it is validation-set or hidden-test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _f(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _tab(caption, label, header, rows, note=None):
    out = ["\\begin{table}[!t]", "\\centering", "\\caption{%s}" % caption,
           "\\label{%s}" % label,
           "\\begin{tabular}{l" + "c" * (len(header) - 1) + "}", "\\hline",
           " & ".join("\\textbf{%s}" % h for h in header) + " \\\\", "\\hline"]
    out += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    out += ["\\hline", "\\end{tabular}"]
    if note:
        out.append("\\\\[2pt]\\footnotesize{%s}" % note)
    out.append("\\end{table}")
    return "\n".join(out)


def _ok(results):
    return [r for r in results if "error" not in r]


def performance_table(results, path=None):
    rows = [[r["model"], _f(r.get("mean_dice_mean")), _f(r.get("mean_iou_mean")),
             _f(r.get("mean_precision_mean")), _f(r.get("mean_recall_mean")),
             _f(r.get("mean_accuracy_mean")), _f(r.get("mAP_mean"))]
            for r in _ok(results)]
    tex = _tab("Segmentation performance on the held-out test split.",
               "tab:performance",
               ["Model", "Dice/F1", "IoU", "Precision", "Recall", "Accuracy", "mAP"],
               rows,
               "Mean over the nested WT, TC and ET regions. Dice and F1 are "
               "identical for binary overlap and reported once. Cases with empty "
               "ground truth for a region are excluded from that region's mean.")
    if path: Path(path).write_text(tex)
    return tex


def region_table(results, path=None):
    rows = [[r["model"], _f(r.get("WT_dice_mean")), _f(r.get("TC_dice_mean")),
             _f(r.get("ET_dice_mean"))] for r in _ok(results)]
    tex = _tab("Per-region Dice similarity coefficient.", "tab:regions",
               ["Model", "WT", "TC", "ET"], rows,
               "These are the regions the BraTS literature reports, so this is "
               "the row that aligns with published work.")
    if path: Path(path).write_text(tex)
    return tex


def efficiency_table(results, path=None):
    rows = [[r["model"], _f(r.get("params_M"), 2), _f(r.get("model_size_MB"), 1),
             r.get("epochs_run", "--"), _f(r.get("mean_epoch_time_s"), 1),
             _f((r.get("total_train_time_s") or 0) / 60, 1),
             _f(r.get("peak_MB"), 0)] for r in _ok(results)]
    dev = next((r.get("device") for r in _ok(results) if r.get("device")), "unspecified")
    tex = _tab("Model complexity and training cost.", "tab:efficiency",
               ["Model", "Params (M)", "Size (MB)", "Epochs", "s/epoch",
                "Total (min)", "Peak VRAM (MB)"], rows,
               "All training performed on %s via Google Colab Pro." % dev)
    if path: Path(path).write_text(tex)
    return tex


def ablation_table(results, factor, path=None):
    rows = [[r["name"].split("-")[-1], _f(r.get("mean_dice_mean")),
             _f(r.get("ET_dice_mean")), _f(r.get("mean_iou_mean")),
             _f(r.get("mean_epoch_time_s"), 1)] for r in _ok(results)]
    tex = _tab("Ablation study: effect of %s." % factor, "tab:abl_%s" % factor,
               [factor.replace("_", " ").capitalize(), "Mean Dice", "ET Dice",
                "IoU", "s/epoch"], rows)
    if path: Path(path).write_text(tex)
    return tex


def data_efficiency_table(results, path=None):
    rows = [[r.get("n_cases", "--"), _f(r.get("mean_dice_mean")),
             _f(r.get("ET_dice_mean")), _f(r.get("mean_epoch_time_s"), 1)]
            for r in _ok(results)]
    tex = _tab("Effect of training-set size on segmentation performance.",
               "tab:data_efficiency",
               ["Training cases", "Mean Dice", "ET Dice", "s/epoch"], rows,
               "Subsets drawn at subject level with a fixed seed, so no subject "
               "is split across train and test.")
    if path: Path(path).write_text(tex)
    return tex


def stratified_table(rows, path=None):
    body = [[r["bin_cm3"], r["n"], _f(r["mean_dice"]), _f(r["std_dice"]),
             r["n_missed"], _f(r["miss_rate"], 3)] for r in rows]
    tex = _tab("Enhancing-tumour performance stratified by lesion volume.",
               "tab:stratified",
               ["Volume (cm$^3$)", "$n$", "Mean Dice", "SD", "Missed", "Miss rate"],
               body,
               "A missed case is one where the model predicts no enhancing tumour "
               "while ground truth contains some.")
    if path: Path(path).write_text(tex)
    return tex


def transfer_table(results, path=None):
    rows = []
    for r in _ok(results):
        cond = "Fine-tuned (pre-op init)" if r.get("pretrained") else "Scratch"
        rows.append([r["model"], cond, _f(r.get("mean_dice_mean")),
                     _f(r.get("WT_dice_mean")), _f(r.get("ET_dice_mean"))])
    tex = _tab("Pre-operative to post-treatment transfer on MU-Glioma-Post.",
               "tab:transfer",
               ["Model", "Initialisation", "Mean Dice", "WT", "ET"], rows,
               "Pre-operative initialisation from published BraTS pretrained "
               "weights; the source training protocol is therefore not under our "
               "control, which we note as a limitation.")
    if path: Path(path).write_text(tex)
    return tex


def comparison_table(ours, path=None):
    lit = [["Ferreira et al.~\\cite{ferreira2024} (BraTS'24 winner)",
            "nnU-Net + SwinUNETR + MedNeXt ens.", "0.8734", "0.7500", "0.7557"],
           ["Roy et al.~\\cite{mednext2023}", "MedNeXt-L k5 (BraTS'21)",
            "0.8801", "--", "--"],
           ["Hatamizadeh et al.~\\cite{swinunetr2022}", "SwinUNETR (BraTS'21)",
            "--", "--", "--"]]
    ours_row = ["\\textbf{This work}", ours.get("model", "--"),
                _f(ours.get("WT_dice_mean")), _f(ours.get("TC_dice_mean")),
                _f(ours.get("ET_dice_mean"))]
    tex = _tab("Performance comparison with published works.", "tab:comparison",
               ["Work", "Method", "WT", "TC", "ET"], lit + [ours_row],
               "VERIFY every external figure against its source before "
               "submission. Reported values use differing evaluation protocols "
               "(validation vs hidden test, full-resolution training on full "
               "cohorts), so they are indicative rather than directly comparable "
               "with our reduced-patch, subset-trained results.")
    if path: Path(path).write_text(tex)
    return tex


def all_tables(results, out_dir, strat_rows=None, transfer_results=None):
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    made = {"performance": performance_table(results, d/"tab_performance.tex"),
            "regions": region_table(results, d/"tab_regions.tex"),
            "efficiency": efficiency_table(results, d/"tab_efficiency.tex")}
    ok = _ok(results)
    if ok:
        best = max(ok, key=lambda r: r.get("mean_dice_mean") or -1)
        made["comparison"] = comparison_table(best, d/"tab_comparison.tex")
        made["best_model"] = best["model"]
    if strat_rows:
        made["stratified"] = stratified_table(strat_rows, d/"tab_stratified.tex")
    if transfer_results:
        made["transfer"] = transfer_table(transfer_results, d/"tab_transfer.tex")
    (d/"results.json").write_text(json.dumps(results, indent=2, default=str))
    print("[tables] wrote %d .tex files -> %s" % (
        len([k for k in made if k != 'best_model']), d))
    return made
