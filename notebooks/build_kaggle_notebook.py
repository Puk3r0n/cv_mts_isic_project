# Сборщик Kaggle-ноутбука: читает src/*.py, configs/baseline.yaml и
# scripts/prepare_isic2019.py, встраивает их как %%writefile-ячейки и добавляет
# шаги запуска пайплайна.
# Запуск из корня проекта:
#   python notebooks/build_kaggle_notebook.py
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "kaggle_pipeline.ipynb")

# Порядок важен: utils/config/dataset/model/metrics/transforms — раньше зависимых.
SRC_MODULES = [
    "src/__init__.py",
    "src/utils.py",
    "src/config.py",
    "src/dataset.py",
    "src/model.py",
    "src/metrics.py",
    "src/transforms.py",
    "src/train.py",
    "src/evaluate.py",
    "src/data_audit.py",
    "src/thresholds.py",
    "src/calibration.py",
    "src/subgroups.py",
    "src/error_cases.py",
    "src/figures.py",
]

BASE = "/kaggle/working/isic_project"


def read(path):
    with open(os.path.join(ROOT, path), "r") as f:
        return f.read()


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def writefile_cell(target_path, content):
    return code_cell("%%writefile " + target_path + "\n" + content)


HEADER_MD = """# ISIC 2019 — прогон бейслайна на Kaggle

Ноутбук встраивает код проекта (`src/`, `configs/`, `scripts/`) напрямую,
ничего качать/клонировать не надо.

## Что нужно сделать один раз в Kaggle UI

1. **Accelerator** → GPU T4 x1 (или P100).
2. **Add Input** → датасет `andrewmvd/isic-2019` (или любой с файлами
   `ISIC_2019_Training_GroundTruth.csv`, `ISIC_2019_Training_Metadata.csv` и
   папкой изображений).
3. **Run All**.

После завершения — скачать `/kaggle/working/isic_outputs.zip` и распаковать
локально в `reports/` + `runs/`.
"""

ENV_CHECK = """import sys, torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
"""

MKDIRS = """import os
for sub in ("src", "scripts", "configs", "data", "runs", "reports"):
    os.makedirs("%s/" % "{base}" + sub, exist_ok=True)
%cd {base}
""".replace("{base}", BASE)

DETECT_DATA = """# автопоиск путей к ISIC 2019 внутри /kaggle/input
import os

print("=== содержимое /kaggle/input ===")
for root, dirs, files in os.walk("/kaggle/input"):
    if root.count("/") - 2 > 3:
        continue
    print(root)
    for f in sorted(files)[:10]:
        print("  ", f)

def find_one(tokens):
    out = []
    for root, _, files in os.walk("/kaggle/input"):
        for name in files:
            low = name.lower()
            if all(t in low for t in tokens):
                out.append(os.path.join(root, name))
    return out

gt_candidates = find_one(["groundtruth", ".csv"]) or find_one(["ground_truth", ".csv"]) or find_one(["labels", ".csv"])
meta_candidates = find_one(["metadata", ".csv"]) or find_one(["meta", ".csv"])
img_candidates = sorted({root for root, _, files in os.walk("/kaggle/input")
                         if len([f for f in files if f.lower().endswith(".jpg")]) > 1000})

print("\\nGT:", gt_candidates)
print("META:", meta_candidates)
print("IMAGES:", img_candidates)
assert gt_candidates, "Не найден файл разметки ISIC 2019 GroundTruth CSV"
assert img_candidates, "Не найдена папка с изображениями (>1000 .jpg)"

GT_PATH, IMG_SRC = gt_candidates[0], img_candidates[0]
META_PATH = meta_candidates[0] if meta_candidates else ""

# симлинк на изображения (не копируем ~9 ГБ)
link = "{base}/data/images"
if os.path.islink(link):
    os.unlink(link)
if not os.path.lexists(link):
    os.symlink(IMG_SRC, link)
print("images ->", os.readlink(link) if os.path.islink(link) else link)

os.environ["ISIC_DATA_DIR"] = "{base}/data"
os.environ["ISIC_RUNS_DIR"] = "{base}/runs"
os.environ["ISIC_REPORTS_DIR"] = "{base}/reports"
""".replace("{base}", BASE)

RUN_PREPARE = """meta_arg = ("--metadata " + META_PATH) if META_PATH else ""
!python scripts/prepare_isic2019.py --gt "$GT_PATH" $meta_arg --out data/ground_truth.csv
"""

RUN_PIPELINE = """# полный пайплайн на едином конфиге
!python -m src.data_audit
!python -m src.train
!python -m src.evaluate
!python -m src.thresholds
!python -m src.calibration
!python -m src.subgroups
!python -m src.error_cases
!python -m src.stress_test
!python -m src.figures
"""

SUMMARIZE = """import json, pandas as pd, os
rep = "{base}/reports"
run = "{base}/runs"

print("=== summary.json ===")
print(json.dumps(json.load(open(os.path.join(rep, "summary.json"))), indent=2, ensure_ascii=False))
print("\\n=== per_class_metrics ===")
print(pd.read_csv(os.path.join(rep, "per_class_metrics.csv")).to_string())
print("\\n=== slice_metrics (worst-slice) ===")
print(pd.read_csv(os.path.join(rep, "slice_metrics.csv")).to_string())
print("\\n=== stress_metrics ===")
print(pd.read_csv(os.path.join(rep, "stress_metrics.csv")).to_string())
print("\\n=== operating_point.json ===")
print(json.dumps(json.load(open(os.path.join(rep, "operating_point.json"))), indent=2, ensure_ascii=False))
""".replace("{base}", BASE)

PACK = """import shutil
shutil.make_archive("/kaggle/working/isic_outputs", "zip", "{base}")
print("готово: /kaggle/working/isic_outputs.zip")
""".replace("{base}", BASE)

FOOTER_MD = """## После прогона

1. Скачать `/kaggle/working/isic_outputs.zip`.
2. Распаковать локально — содержимое `reports/` и `runs/` заменит заготовки.
3. Сверить числа в `reports/*.md` с обновлёнными `summary.json`,
   `threshold_sweep.csv`, `calibration.json`.
"""


def build():
    cells = [md_cell(HEADER_MD), code_cell(ENV_CHECK), code_cell(MKDIRS)]
    cells.append(writefile_cell(BASE + "/configs/baseline.yaml", read("configs/baseline.yaml")))
    for module in SRC_MODULES:
        cells.append(writefile_cell(BASE + "/" + module, read(module)))
    cells.append(writefile_cell(BASE + "/scripts/prepare_isic2019.py", read("scripts/prepare_isic2019.py")))
    cells += [
        code_cell(DETECT_DATA),
        code_cell(RUN_PREPARE),
        code_cell(RUN_PIPELINE),
        code_cell(SUMMARIZE),
        code_cell(PACK),
        md_cell(FOOTER_MD),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(OUT, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", OUT, "(" + str(len(cells)) + " cells)")


if __name__ == "__main__":
    build()
