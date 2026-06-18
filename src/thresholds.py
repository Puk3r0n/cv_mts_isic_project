import csv
import os
from pathlib import Path

import click
import numpy as np

from src.config import load_config
from src.metrics import sweep, threshold_grid, uniform_risk
from src.utils import dump_json


def load_probs(reports_dir):
    data = np.load(os.path.join(reports_dir, "val_probs.npz"))
    return data["probs"], data["labels"]


def write_sweep_csv(rows, path):
    cols = ["threshold", "tp", "fp", "fn", "tn", "recall", "precision", "specificity", "fpr", "f1", "risk"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in cols})


def run_thresholds(cfg):
    reports_dir = cfg["paths"]["reports_dir"]
    probs, labels = load_probs(reports_dir)

    decision = cfg["decision"]
    taus = threshold_grid(decision["threshold_step"])
    c_fp, c_fn = decision["cost_fp"], decision["cost_fn"]

    pos_idx = decision["positive_index"]
    pos_scores = probs[:, pos_idx]
    positive = (labels == pos_idx).astype(int)
    rows = sweep(pos_scores, positive, taus, c_fp, c_fn)
    write_sweep_csv(rows, os.path.join(reports_dir, "threshold_sweep.csv"))

    best = min(rows, key=lambda r: r["risk"])
    default = next((r for r in rows if abs(r["threshold"] - decision["default_threshold"]) < 1e-9), None)
    default_risk = uniform_risk(default, c_fp, c_fn) if default else None

    mal_idx = decision["malignant_indices"]
    mal_scores = probs[:, mal_idx].sum(axis=1)
    mal_positive = np.isin(labels, mal_idx).astype(int)
    mal_rows = sweep(mal_scores, mal_positive, taus, c_fp, c_fn)
    mal_best = min(mal_rows, key=lambda r: r["risk"])

    gain = round(best["risk"] / default_risk - 1, 4) if default_risk else None
    payload = {
        "positive_class": decision["positive_class"],
        "cost_fp": c_fp,
        "cost_fn": c_fn,
        "operating_point": best,
        "default_threshold": decision["default_threshold"],
        "default_point": default,
        "default_risk": default_risk,
        "risk_gain_vs_default": gain,
        "rows": rows,
        "malignant_sensitivity": {"classes": decision["malignant_classes"],
                                  "operating_point": mal_best, "rows": mal_rows},
    }
    dump_json(os.path.join(reports_dir, "threshold_sweep.json"), payload)
    dump_json(os.path.join(reports_dir, "operating_point.json"), {
        "positive_class": decision["positive_class"], "cost_fp": c_fp, "cost_fn": c_fn,
        "operating_point": best, "default_point": default, "risk_gain_vs_default": gain})

    print("operating tau=%.2f recall=%.3f precision=%.3f risk=%d (gain vs default: %s)"
          % (best["threshold"], best["recall"], best["precision"], best["risk"], gain))
    return Path(reports_dir) / "operating_point.json"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", type=click.Path(path_type=Path), default=None)
def main(config):
    click.echo(run_thresholds(load_config(config)))


if __name__ == "__main__":
    main()
