import os
from pathlib import Path

import click
import pandas as pd
from sklearn.metrics import f1_score

from src.config import load_config
from src.dataset import load_table


def slice_metrics(sub, pos_idx, n_classes):
    y_true = sub["label"].to_numpy()
    y_pred = sub["pred"].to_numpy()
    acc = float((y_true == y_pred).mean()) if len(sub) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(n_classes)),
                              average="macro", zero_division=0)) if len(sub) else 0.0
    pos_mask = y_true == pos_idx
    pos_support = int(pos_mask.sum())
    pos_recall = float((y_pred[pos_mask] == pos_idx).mean()) if pos_support else None
    return {"support": len(sub), "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "pos_support": pos_support, "pos_recall": round(pos_recall, 4) if pos_recall is not None else None}


def build_slices(index, classes, decision):
    pos_idx = decision["positive_index"]
    mal_idx = set(decision["malignant_indices"])
    nc = len(classes)
    rows = []

    for i, name in enumerate(classes):
        sub = index[index["label"] == i]
        if len(sub):
            rows.append({"slice": f"class:{name}", **slice_metrics(sub, pos_idx, nc)})

    for label, mask in (("group:malignant", index["label"].isin(mal_idx)),
                        ("group:benign", ~index["label"].isin(mal_idx))):
        sub = index[mask]
        if len(sub):
            rows.append({"slice": label, **slice_metrics(sub, pos_idx, nc)})

    bands = [("conf:low(<0.5)", index["max_prob"] < 0.5),
             ("conf:mid(0.5-0.8)", (index["max_prob"] >= 0.5) & (index["max_prob"] < 0.8)),
             ("conf:high(>=0.8)", index["max_prob"] >= 0.8)]
    for label, mask in bands:
        sub = index[mask]
        if len(sub):
            rows.append({"slice": label, **slice_metrics(sub, pos_idx, nc)})

    if "group_size" in index.columns:
        for label, mask in (("lesion:singleton", index["group_size"] == 1),
                            ("lesion:multi_frame", index["group_size"] > 1)):
            sub = index[mask]
            if len(sub):
                rows.append({"slice": label, **slice_metrics(sub, pos_idx, nc)})

    return rows


def run_subgroups(cfg):
    reports_dir = cfg["paths"]["reports_dir"]
    index = pd.read_csv(os.path.join(reports_dir, "val_index.csv"))

    gt = load_table(cfg["paths"]["ground_truth"])
    group_by = cfg["split"]["group_by"]
    if group_by in gt.columns:
        sizes = gt[group_by].astype(str).value_counts()
        name_to_group = dict(zip(gt["image"].astype(str), gt[group_by].astype(str)))
        index["group_size"] = index["image"].astype(str).map(name_to_group).map(sizes).fillna(1).astype(int)

    rows = build_slices(index, cfg["classes"], cfg["decision"])
    rows.sort(key=lambda r: r["accuracy"])

    out = os.path.join(reports_dir, "slice_metrics.csv")
    pd.DataFrame(rows, columns=["slice", "support", "accuracy", "macro_f1", "pos_support", "pos_recall"]).to_csv(
        out, index=False)
    print("worst slices:", [(r["slice"], r["accuracy"]) for r in rows[:3]])
    click.echo(out)
    return out


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", type=click.Path(path_type=Path), default=None)
def main(config):
    run_subgroups(load_config(config))


if __name__ == "__main__":
    main()
