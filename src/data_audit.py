import os
from collections import Counter
from pathlib import Path

import click

from src.config import load_config
from src.dataset import load_table, split_frame
from src.utils import dump_json


def group_overlap(train_df, val_df, group_by):
    if group_by not in train_df.columns:
        return {"available": False}
    train_groups = set(train_df[group_by].astype(str))
    val_groups = set(val_df[group_by].astype(str))
    overlap = train_groups & val_groups
    return {
        "available": True,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "overlapping_groups": len(overlap),
        "overlap_examples": sorted(overlap)[:10],
    }


def collect_stats(cfg):
    classes = cfg["classes"]
    decision = cfg["decision"]
    frame = load_table(cfg["paths"]["ground_truth"])

    label_counts = Counter(int(x) for x in frame["label"])
    class_counts = {name: label_counts.get(i, 0) for i, name in enumerate(classes)}
    total = len(frame)

    group_by = cfg["split"]["group_by"]
    if group_by in frame.columns:
        group_sizes = Counter(frame[group_by].astype(str))
        n_groups = len(group_sizes)
        multi = sum(1 for v in group_sizes.values() if v > 1)
        max_group = max(group_sizes.values()) if group_sizes else 0
    else:
        n_groups, multi, max_group = total, 0, 1

    train_df, val_df = split_frame(frame, group_by, cfg["split"]["val_size"], cfg["split"]["random_state"])
    malignant = decision["malignant_classes"]
    malignant_count = sum(class_counts[c] for c in malignant)
    ordered = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "ground_truth": cfg["paths"]["ground_truth"],
        "n_rows": total,
        "n_classes": len(classes),
        "class_counts": class_counts,
        "class_fractions": {k: round(v / max(1, total), 4) for k, v in class_counts.items()},
        "dominant_class": ordered[0][0] if ordered else None,
        "rarest_class": ordered[-1][0] if ordered else None,
        "imbalance_ratio": round(ordered[0][1] / max(1, ordered[-1][1]), 2) if ordered else None,
        "malignant_classes": malignant,
        "malignant_fraction": round(malignant_count / max(1, total), 4),
        "grouping": {
            "group_by": group_by,
            "n_groups": n_groups,
            "multi_frame_groups": multi,
            "max_group_size": max_group,
            "avg_frames_per_group": round(total / max(1, n_groups), 3),
        },
        "split": {
            "val_size": cfg["split"]["val_size"],
            "random_state": cfg["split"]["random_state"],
            "n_train": len(train_df),
            "n_val": len(val_df),
            "train_class_counts": {name: int((train_df["label"] == i).sum()) for i, name in enumerate(classes)},
            "val_class_counts": {name: int((val_df["label"] == i).sum()) for i, name in enumerate(classes)},
        },
        "group_overlap": group_overlap(train_df, val_df, group_by),
    }


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", type=click.Path(path_type=Path), default=None)
def main(config):
    cfg = load_config(config)
    stats = collect_stats(cfg)
    out = dump_json(os.path.join(cfg["paths"]["reports_dir"], "data_audit.json"), stats)
    print("rows=%d classes=%d dominant=%s imbalance=%.1fx groups=%d overlap=%s"
          % (stats["n_rows"], stats["n_classes"], stats["dominant_class"], stats["imbalance_ratio"] or 0,
             stats["grouping"]["n_groups"], stats["group_overlap"].get("overlapping_groups")))
    click.echo(out)


if __name__ == "__main__":
    main()
