import os
import random
from pathlib import Path

import click
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.config import load_config
from src.utils import dump_json, ensure_dir, load_json

GREEN, RED = "#20a050", "#d04040"


def resolve_image(images_dir, name):
    for suffix in (".jpg", ".jpeg", ".png"):
        path = os.path.join(images_dir, name + suffix)
        if os.path.isfile(path):
            return path
    return os.path.join(images_dir, name)


def annotate(image_path, lines, color):
    image = Image.open(image_path).convert("RGB")
    if max(image.size) < 256:
        scale = 256 // max(1, min(image.size)) + 1
        image = image.resize((image.width * scale, image.height * scale))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    y = 2
    for text in lines:
        box = draw.textbbox((4, y), text, font=font)
        draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=color)
        draw.text((4, y), text, fill="white", font=font)
        y = box[3] + 3
    return image


def operating_tau(cfg):
    op_path = os.path.join(cfg["paths"]["reports_dir"], "operating_point.json")
    if os.path.isfile(op_path):
        return float(load_json(op_path)["operating_point"]["threshold"])
    return float(cfg["decision"]["default_threshold"])


def run_error_cases(cfg, per_category=12):
    classes = cfg["classes"]
    reports_dir = cfg["paths"]["reports_dir"]
    images_dir = cfg["paths"]["images_dir"]
    pos_idx = cfg["decision"]["positive_index"]
    pos_name = cfg["decision"]["positive_class"]

    index = pd.read_csv(os.path.join(reports_dir, "val_index.csv"))
    pos_scores = np.load(os.path.join(reports_dir, "val_probs.npz"))["probs"][:, pos_idx]
    tau = operating_tau(cfg)

    buckets = {"tp": [], "fp": [], "fn": [], "tn": []}
    for i, row in index.iterrows():
        is_pos = int(row["label"]) == pos_idx
        emitted = pos_scores[i] >= tau
        kind = "tp" if emitted and is_pos else "fp" if emitted else "fn" if is_pos else "tn"
        buckets[kind].append({"image": str(row["image"]), "true": classes[int(row["label"])],
                              "pred": classes[int(row["pred"])], "score": float(pos_scores[i])})

    rng = random.Random(cfg["seed"])
    out_root = ensure_dir(os.path.join(reports_dir, "error_cases"))
    selection = {}
    for kind, items in buckets.items():
        cat_dir = ensure_dir(os.path.join(out_root, kind))
        for old in Path(cat_dir).glob("*.jpg"):
            old.unlink()
        ordered = sorted(items, key=lambda c: c["score"], reverse=(kind != "fn"))
        chosen = ordered[:per_category // 2]
        rest = [c for c in items if c not in chosen]
        if rest:
            chosen += rng.sample(rest, min(per_category - per_category // 2, len(rest)))
        color = GREEN if kind in ("tp", "tn") else RED
        for c in chosen:
            lines = [f"{kind.upper()}  {pos_name}={c['score']:.2f}", f"true={c['true']} pred={c['pred']}"]
            annotate(resolve_image(images_dir, c["image"]), lines, color).save(
                os.path.join(cat_dir, c["image"] + ".jpg"))
        selection[kind] = {"available": len(items), "selected": [c["image"] for c in chosen]}

    dump_json(os.path.join(out_root, "selection.json"),
              {"tau": tau, "positive_class": pos_name, "per_category": per_category, "categories": selection})
    print("error cases @tau=%.2f:" % tau, {k: selection[k]["available"] for k in buckets})
    return Path(out_root)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", type=click.Path(path_type=Path), default=None)
@click.option("--per-category", type=int, default=12)
def main(config, per_category):
    click.echo(run_error_cases(load_config(config), per_category))


if __name__ == "__main__":
    main()
