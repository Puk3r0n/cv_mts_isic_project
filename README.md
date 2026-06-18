# ISIC 2019 — бейслайн классификации дерматоскопии

Восьмиклассовая классификация поражений кожи (ResNet18) на данных ISIC 2019.
Постановка задачи — `PROJECT_PASSPORT.md`, полный отчёт — `reports/final_report.md`.

## Установка

```bash
pip install -r requirements.txt
```

## Данные

Демо-датасет для проверки пайплайна:

```bash
python scripts/bootstrap_demo_dataset.py
```

Полный ISIC 2019 (изображения в `data/images/`, разметка из челленджа):

```bash
python scripts/prepare_isic2019.py \
  --gt ISIC_2019_Training_GroundTruth.csv \
  --metadata ISIC_2019_Training_Metadata.csv \
  --out data/ground_truth.csv
```

Прогон на полном наборе делался на Kaggle (датасет `andrewmvd/isic-2019`),
ноутбук — `notebooks/kaggle_pipeline.ipynb`.

## Обучение

```bash
python -m src.train --config configs/baseline.yaml
```

Чекпойнт и кривые обучения пишутся в `runs/`.

## Оценка и аналитика

```bash
python -m src.data_audit      # аудит данных и протокола
python -m src.evaluate        # per-class, confusion, ROC, summary
python -m src.thresholds      # threshold sweep + operating point
python -m src.calibration     # ECE + reliability
python -m src.subgroups       # worst-slice по подгруппам
python -m src.error_cases     # картинки ошибок по меланоме
python -m src.figures         # графики
```

Все артефакты сохраняются в `reports/`.

## Стресс-тест

```bash
python -m src.stress_test --config configs/baseline.yaml
```

Любой шаг принимает `--config`; пути можно переопределить переменными окружения
`ISIC_DATA_DIR`, `ISIC_RUNS_DIR`, `ISIC_REPORTS_DIR`.
