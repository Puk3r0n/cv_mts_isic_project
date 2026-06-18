import argparse
import os
import sys

import pandas as pd

CLASS_ORDER = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="ISIC_2019_Training_GroundTruth.csv")
    parser.add_argument("--metadata", default=None, help="ISIC_2019_Training_Metadata.csv (опционально)")
    parser.add_argument("--out", required=True, help="путь к итоговому ground_truth.csv")
    args = parser.parse_args()

    gt = pd.read_csv(args.gt)
    missing = [c for c in CLASS_ORDER if c not in gt.columns]
    if missing:
        print("в GT отсутствуют колонки:", missing, file=sys.stderr)
        sys.exit(1)

    if "UNK" in gt.columns:
        gt = gt[gt["UNK"] < 0.5].copy()

    labels = gt[CLASS_ORDER].to_numpy().argmax(axis=1)
    out = pd.DataFrame({"image": gt["image"].astype(str), "label": labels})

    if args.metadata and os.path.isfile(args.metadata):
        meta = pd.read_csv(args.metadata)
        if "lesion_id" in meta.columns:
            out = out.merge(meta[["image", "lesion_id"]], on="image", how="left")
        else:
            print("в metadata нет lesion_id — сплит будет по image", file=sys.stderr)
    else:
        print("metadata не передана — сплит будет по image", file=sys.stderr)

    if "lesion_id" not in out.columns:
        out["lesion_id"] = out["image"]
    else:
        out["lesion_id"] = out["lesion_id"].fillna(out["image"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    print("saved:", args.out)
    print("n_rows:", len(out))
    print("распределение классов:")
    counts = out["label"].value_counts().sort_index()
    for i, name in enumerate(CLASS_ORDER):
        print("  %d %-5s %6d" % (i, name, int(counts.get(i, 0))))
    n_groups = out["lesion_id"].nunique()
    print("уникальных lesion_id:", n_groups)


if __name__ == "__main__":
    main()
