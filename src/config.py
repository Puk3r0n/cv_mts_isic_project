import os
from pathlib import Path

import yaml

from src.utils import PROJECT_ROOT, as_path

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"


def load_config(config_path=None):
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["config_path"] = str(config_path.resolve())

    paths = cfg["paths"]
    paths["data_dir"] = str(as_path(os.environ.get("ISIC_DATA_DIR", paths["data_dir"])))
    paths["runs_dir"] = str(as_path(os.environ.get("ISIC_RUNS_DIR", paths["runs_dir"])))
    paths["reports_dir"] = str(as_path(os.environ.get("ISIC_REPORTS_DIR", paths["reports_dir"])))
    paths["images_dir"] = str(as_path(paths["images_dir"], paths["data_dir"]))
    paths["ground_truth"] = str(as_path(paths["ground_truth"], paths["data_dir"]))
    paths["checkpoint"] = str(as_path(paths["checkpoint"], paths["runs_dir"]))

    classes = cfg["classes"]
    d = cfg["decision"]
    d["positive_index"] = classes.index(d["positive_class"])
    d["malignant_indices"] = [classes.index(c) for c in d["malignant_classes"]]
    return cfg


def num_classes(cfg):
    return len(cfg["classes"])
