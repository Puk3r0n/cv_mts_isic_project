import json
import os
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config
from src.utils import ensure_dir


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def training_curves(runs_dir, out):
    history_path = os.path.join(runs_dir, "history.json")
    if not os.path.isfile(history_path):
        return
    hist = json.loads(Path(history_path).read_text())
    epochs = [h["epoch"] for h in hist]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_loss.plot(epochs, [h["train_loss"] for h in hist], marker="o", label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in hist], marker="o", label="val")
    ax_loss.set_title("Loss по эпохам"); ax_loss.set_xlabel("эпоха"); ax_loss.set_ylabel("loss")
    ax_loss.grid(alpha=0.3); ax_loss.legend()
    ax_acc.plot(epochs, [h["train_acc"] for h in hist], marker="o", label="train")
    ax_acc.plot(epochs, [h["val_acc"] for h in hist], marker="o", label="val")
    ax_acc.set_title("Accuracy по эпохам"); ax_acc.set_xlabel("эпоха"); ax_acc.set_ylabel("accuracy")
    ax_acc.grid(alpha=0.3); ax_acc.legend()
    save(fig, out)


def roc_curve_fig(reports_dir, positive_class, out):
    path = os.path.join(reports_dir, "roc_%s.csv" % positive_class.lower())
    if not os.path.isfile(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(df["fpr"], df["tpr"], linewidth=1.6)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR (recall)")
    ax.set_title(f"ROC: {positive_class} vs всё")
    ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    save(fig, out)


def load_sweep(reports_dir):
    path = os.path.join(reports_dir, "threshold_sweep.csv")
    return pd.read_csv(path) if os.path.isfile(path) else None


def precision_recall_fig(reports_dir, out):
    df = load_sweep(reports_dir)
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(df["recall"], df["precision"], marker=".", linewidth=1.3)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision vs recall (по порогу)")
    ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    save(fig, out)


def metrics_vs_threshold_fig(reports_dir, out):
    df = load_sweep(reports_dir)
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for col in ("precision", "recall", "f1", "specificity"):
        if col in df.columns:
            ax.plot(df["threshold"], df[col], linewidth=1.4, label=col)
    ax.set_xlabel("порог tau"); ax.set_ylabel("значение метрики")
    ax.set_title("Метрики по порогу"); ax.grid(alpha=0.3); ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    save(fig, out)


def risk_vs_threshold_fig(reports_dir, out):
    df = load_sweep(reports_dir)
    if df is None:
        return
    tau = df["threshold"].tolist()
    risk = df["risk"].tolist()
    argmin = tau[int(np.argmin(risk))]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(tau, risk, marker=".", color="C3", linewidth=1.4)
    ax.axvline(argmin, color="C0", linestyle="--", linewidth=0.9)
    ax.text(argmin, max(risk) * 0.95, f"  argmin tau={argmin}", color="C0")
    ax.set_xlabel("порог tau"); ax.set_ylabel("risk = c_fp·FP + c_fn·FN")
    ax.set_title("Функция риска по порогу"); ax.grid(alpha=0.3); ax.set_xlim(0, 1)
    save(fig, out)


def confusion_fig(reports_dir, out):
    path = os.path.join(reports_dir, "confusion_matrix_normalized.csv")
    if not os.path.isfile(path):
        return
    df = pd.read_csv(path, index_col=0)
    mat = df.to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index)
    ax.set_xlabel("предсказание"); ax.set_ylabel("истина")
    ax.set_title("Confusion matrix (нормированная по строкам)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] >= 0.01:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, out)


def stress_fig(reports_dir, out):
    path = os.path.join(reports_dir, "stress_metrics.csv")
    if not os.path.isfile(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    if "scenario" in df.columns:
        ax.bar(df["scenario"], df["macro_f1"], alpha=0.85)
        ax.set_ylabel("macro-F1"); ax.set_title("Стресс-сценарии: macro-F1")
        ax.set_xticks(range(len(df))); ax.set_xticklabels(df["scenario"], rotation=30, ha="right")
    else:
        for name, grp in df.groupby("perturbation"):
            ax.plot(grp["alpha"], grp["macro_f1"], marker="o", label=name)
        ax.set_xlabel("alpha"); ax.set_ylabel("macro-F1")
        ax.set_title("Деградация macro-F1 по искажениям"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, out)


def reliability_fig(reports_dir, out):
    path = os.path.join(reports_dir, "calibration.json")
    if not os.path.isfile(path):
        return
    data = json.loads(Path(path).read_text())
    bins = [b for b in data["bins"] if b["count"] > 0]
    if not bins:
        return
    centers = [(b["low"] + b["high"]) / 2 for b in bins]
    width = (bins[0]["high"] - bins[0]["low"]) * 0.9
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(centers, [b["acc"] for b in bins], width=width, alpha=0.7, edgecolor="black", label="accuracy")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, label="идеальная калибровка")
    ax.scatter(centers, [b["conf"] for b in bins], color="red", marker="x", s=50, zorder=3, label="ср. уверенность")
    ax.set_xlabel("предсказанная уверенность"); ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability diagram (ECE = {data['ece']:.3f})")
    ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.legend(loc="upper left")
    save(fig, out)


def run_figures(cfg):
    reports_dir = cfg["paths"]["reports_dir"]
    runs_dir = cfg["paths"]["runs_dir"]
    positive_class = cfg["decision"]["positive_class"]
    figures_dir = ensure_dir(os.path.join(reports_dir, "figures"))

    training_curves(runs_dir, os.path.join(figures_dir, "training_curves.png"))
    roc_curve_fig(reports_dir, positive_class, os.path.join(figures_dir, "roc_%s.png" % positive_class.lower()))
    precision_recall_fig(reports_dir, os.path.join(figures_dir, "precision_recall.png"))
    metrics_vs_threshold_fig(reports_dir, os.path.join(figures_dir, "metrics_vs_threshold.png"))
    risk_vs_threshold_fig(reports_dir, os.path.join(figures_dir, "risk_vs_threshold.png"))
    confusion_fig(reports_dir, os.path.join(figures_dir, "confusion_matrix.png"))
    stress_fig(reports_dir, os.path.join(figures_dir, "stress_degradation.png"))
    reliability_fig(reports_dir, os.path.join(figures_dir, "reliability_diagram.png"))
    return Path(figures_dir)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", type=click.Path(path_type=Path), default=None)
def main(config):
    click.echo(run_figures(load_config(config)))


if __name__ == "__main__":
    main()
