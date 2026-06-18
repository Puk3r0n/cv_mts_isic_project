import os

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    img_dir = os.path.join(ROOT, "data", "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    for label in range(8):
        for j in range(4):
            lesion = "L%d_%d" % (label, j)
            for k in range(2):
                name = "DEMO_%d_%d_%d" % (label, j, k)
                arr = (rng.integers(0, 255, (64, 64, 3))).astype("uint8")
                arr[:, :, 0] = (arr[:, :, 0] + label * 20) % 255
                Image.fromarray(arr).save(os.path.join(img_dir, name + ".jpg"))
                rows.append({"image": name, "label": label, "lesion_id": lesion})
    csv_path = os.path.join(ROOT, "data", "ground_truth.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("wrote", csv_path, "and jpgs in", img_dir)


if __name__ == "__main__":
    main()
