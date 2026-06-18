import os
from pathlib import Path

import click
import numpy as np

from src.config import load_config
from src.metrics import reliability_bins
from src.utils import dump_json


def run_calibration(cfg):
    reports_dir = cfg["paths"]["reports_dir"]
    data = np.load(os.path.join(reports_dir, "val_probs.npz"))
    probs, labels = data["probs"], data["labels"]
    n_bins = cfg["calibration"]["n_bins"]

    preds = probs.argmax(axis=1)
    max_conf = probs.max(axis=1)
    multiclass = reliability_bins(max_conf, (preds == labels).astype(int), n_bins)

    pos_idx = cfg["decision"]["positive_index"]
    binary = reliability_bins(probs[:, pos_idx], (labels == pos_idx).astype(int), n_bins)

    payload = {
        "n_bins": n_bins,
        "positive_class": cfg["decision"]["positive_class"],
        "multiclass": multiclass,
        "binary_positive": binary,
        "ece": multiclass["ece"],
        "bins": multiclass["bins"],
    }
    out = dump_json(os.path.join(reports_dir, "calibration.json"), payload)
    print("ECE multiclass=%.4f  ECE %s=%.4f" % (multiclass["ece"], cfg["decision"]["positive_class"], binary["ece"]))
    click.echo(out)
    return out


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", type=click.Path(path_type=Path), default=None)
def main(config):
    run_calibration(load_config(config))


if __name__ == "__main__":
    main()
