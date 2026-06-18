import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import load_config, num_classes
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, get_splits, pick_device
from src.metrics import macro_f1_from_confusion
from src.model import build_model, load_checkpoint
from src.transforms import INVARIANT_RANGES, PERTURBATIONS
from src.utils import dump_json, ensure_dir


class PerturbedDataset(Dataset):
    def __init__(self, frame, images_dir, perturb_fn, alpha, image_size):
        self.frame = frame.reset_index(drop=True)
        self.images_dir = images_dir
        self.perturb_fn = perturb_fn
        self.alpha = alpha
        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.frame)

    def _path(self, name):
        for suffix in (".jpg", ".jpeg", ".png"):
            path = os.path.join(self.images_dir, name + suffix)
            if os.path.isfile(path):
                return path
        return os.path.join(self.images_dir, name)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(self._path(str(row["image"]))).convert("RGB")
        image = self.perturb_fn(image, self.alpha)
        return self.to_tensor(image), int(row["label"])


def evaluate_loader(model, loader, device, nc, pos_idx):
    model.eval()
    confusion = np.zeros((nc, nc), dtype=np.int64)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
            labels = labels.numpy()
            for t, p in zip(labels, preds):
                confusion[t, p] += 1
            correct += int((preds == labels).sum())
            total += len(labels)
    accuracy = correct / max(1, total)
    macro_f1 = macro_f1_from_confusion(confusion)
    support = int(confusion[pos_idx, :].sum())
    pos_recall = int(confusion[pos_idx, pos_idx]) / max(1, support)
    return accuracy, macro_f1, pos_recall


def run_stress(cfg, only=None):
    device = pick_device()
    print("device:", device)

    nc = num_classes(cfg)
    pos_idx = cfg["decision"]["positive_index"]
    image_size = cfg["model"]["image_size"]
    reports_dir = ensure_dir(cfg["paths"]["reports_dir"])

    _, val_df = get_splits(cfg)
    subset = cfg["stress"].get("subset")
    if subset and subset < len(val_df):
        val_df = val_df.sample(n=subset, random_state=cfg["stress"].get("seed", 42)).reset_index(drop=True)

    model = build_model(cfg["model"]["backbone"], nc, pretrained=False).to(device)
    load_checkpoint(model, cfg["paths"]["checkpoint"], device)

    families = PERTURBATIONS if only is None else {only: PERTURBATIONS[only]}
    rows = []
    for name, (fn, grid, unit, identity) in families.items():
        for alpha in grid:
            loader = DataLoader(PerturbedDataset(val_df, cfg["paths"]["images_dir"], fn, alpha, image_size),
                                batch_size=cfg["eval"]["batch_size"], shuffle=False,
                                num_workers=cfg["eval"]["num_workers"], pin_memory=(device.type == "cuda"))
            acc, macro_f1, pos_recall = evaluate_loader(model, loader, device, nc, pos_idx)
            rows.append({"perturbation": name, "alpha": alpha, "unit": unit,
                         "is_identity": alpha == identity, "accuracy": acc,
                         "macro_f1": macro_f1, "pos_recall": pos_recall})
            print("%-12s alpha=%-6s acc=%.4f macro_f1=%.4f pos_recall=%.3f" % (name, alpha, acc, macro_f1, pos_recall))

    pd.DataFrame(rows).to_csv(os.path.join(reports_dir, "stress_metrics.csv"), index=False)
    dump_json(os.path.join(reports_dir, "stress_summary.json"), build_summary(rows, families, cfg))
    print("wrote stress_metrics.csv / stress_summary.json")
    return Path(reports_dir) / "stress_summary.json"


def build_summary(rows, families, cfg):
    summary = {}
    for name in families:
        family_rows = sorted([r for r in rows if r["perturbation"] == name], key=lambda r: r["alpha"])
        nominal = next((r for r in family_rows if r["is_identity"]), family_rows[0])
        worst = min(family_rows, key=lambda r: r["macro_f1"])
        invariance = None
        if name in INVARIANT_RANGES:
            lo, hi = INVARIANT_RANGES[name]
            inside = [r for r in family_rows if lo <= r["alpha"] <= hi]
            max_dev = max((abs(nominal["macro_f1"] - r["macro_f1"]) for r in inside), default=0.0)
            invariance = {"range": [lo, hi], "max_macro_f1_deviation": round(max_dev, 4), "ok": max_dev <= 0.05}
        summary[name] = {
            "unit": family_rows[0]["unit"],
            "nominal_macro_f1": round(nominal["macro_f1"], 4),
            "worst_alpha": worst["alpha"],
            "worst_macro_f1": round(worst["macro_f1"], 4),
            "macro_f1_drop": round(nominal["macro_f1"] - worst["macro_f1"], 4),
            "invariance": invariance,
            "curve": [{"alpha": r["alpha"], "accuracy": round(r["accuracy"], 4),
                       "macro_f1": round(r["macro_f1"], 4), "pos_recall": round(r["pos_recall"], 4)}
                      for r in family_rows],
        }
    return {"positive_class": cfg["decision"]["positive_class"], "families": summary}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", type=click.Path(path_type=Path), default=None)
@click.option("--perturbation", default=None)
@click.option("--subset", type=int, default=None)
def main(config, perturbation, subset):
    cfg = load_config(config)
    if subset is not None:
        cfg["stress"]["subset"] = subset
    click.echo(run_stress(cfg, only=perturbation))


if __name__ == "__main__":
    main()
