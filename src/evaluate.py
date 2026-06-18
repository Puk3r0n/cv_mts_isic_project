import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.config import load_config, num_classes
from src.dataset import ISICDataset, eval_transform, get_splits, pick_device
from src.metrics import sweep, threshold_grid, uniform_risk
from src.model import build_model, load_checkpoint
from src.utils import dump_json, ensure_dir


def predict_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            outputs = model(images.to(device))
            probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.vstack(probs), np.concatenate(labels)


def operating_point(scores, positive, decision):
    rows = sweep(scores, positive, threshold_grid(decision["threshold_step"]),
                 decision["cost_fp"], decision["cost_fn"])
    best = min(rows, key=lambda r: r["risk"])
    default = next((r for r in rows if abs(r["threshold"] - decision["default_threshold"]) < 1e-9), None)
    return best, default, rows


def run_evaluation(cfg):
    device = pick_device()
    print("device:", device)

    classes = cfg["classes"]
    nc = num_classes(cfg)
    reports_dir = ensure_dir(cfg["paths"]["reports_dir"])

    _, val_df = get_splits(cfg)
    loader = DataLoader(ISICDataset(val_df, cfg["paths"]["images_dir"], eval_transform(cfg["model"]["image_size"])),
                        batch_size=cfg["eval"]["batch_size"], shuffle=False,
                        num_workers=cfg["eval"]["num_workers"], pin_memory=(device.type == "cuda"))

    model = build_model(cfg["model"]["backbone"], nc, pretrained=False).to(device)
    load_checkpoint(model, cfg["paths"]["checkpoint"], device)

    probs, labels = predict_probs(model, loader, device)
    preds = probs.argmax(axis=1)

    np.savez(os.path.join(reports_dir, "val_probs.npz"), probs=probs, labels=labels)
    val_df.assign(pred=preds, max_prob=probs.max(axis=1))[["image", "label", "pred", "max_prob"]].to_csv(
        os.path.join(reports_dir, "val_index.csv"), index=False)

    report = classification_report(labels, preds, labels=list(range(nc)),
                                   target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(os.path.join(reports_dir, "per_class_metrics.csv"))

    cm = confusion_matrix(labels, preds, labels=list(range(nc)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(reports_dir, "confusion_matrix.csv"))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums > 0, out=np.zeros_like(cm, dtype=float))
    pd.DataFrame(cm_norm, index=classes, columns=classes).to_csv(
        os.path.join(reports_dir, "confusion_matrix_normalized.csv"))

    decision = cfg["decision"]
    pos_idx = decision["positive_index"]
    scores = probs[:, pos_idx]
    positive = (labels == pos_idx).astype(int)

    roc_auc = float(roc_auc_score(positive, scores)) if positive.sum() else float("nan")
    fpr, tpr, thr = roc_curve(positive, scores)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
        os.path.join(reports_dir, "roc_%s.csv" % decision["positive_class"].lower()), index=False)

    best, default, _ = operating_point(scores, positive, decision)
    default_risk = uniform_risk(default, decision["cost_fp"], decision["cost_fn"]) if default else None

    summary = {
        "positive_class": decision["positive_class"],
        "overall_accuracy": float((preds == labels).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "roc_auc": roc_auc,
        "operating_point": {**best, "cost_fp": decision["cost_fp"], "cost_fn": decision["cost_fn"]},
        "default_threshold": decision["default_threshold"],
        "default_risk": default_risk,
        "risk_gain_vs_default": (round(best["risk"] / default_risk - 1, 4) if default_risk else None),
        "n_val": int(len(labels)),
        "n_positive_val": int(positive.sum()),
    }
    dump_json(os.path.join(reports_dir, "summary.json"), summary)

    print("accuracy=%.3f macro_f1=%.3f roc_auc=%.3f op_tau=%.2f risk=%d"
          % (summary["overall_accuracy"], summary["macro_f1"], roc_auc, best["threshold"], best["risk"]))
    return Path(reports_dir) / "summary.json"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", type=click.Path(path_type=Path), default=None)
def main(config):
    click.echo(run_evaluation(load_config(config)))


if __name__ == "__main__":
    main()
