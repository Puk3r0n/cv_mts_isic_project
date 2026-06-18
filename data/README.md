# Данные

- **`ground_truth.csv`** — таблица для обучения и оценки. Минимальные колонки:
  - `image` — идентификатор файла **без** расширения (ожидается `.jpg` в `images/`).
  - `label` — целое 0..7 в порядке классов MEL, NV, BCC, AK, BKL, DF, VASC, SCC.
  - `lesion_id` — строка или число; **одна и та же** для всех кадров одного поражения (для группового сплита). Если нет уникальных поражений на кадр, можно дублировать `image`.
  - `patient_id` — опционально, для отчётов и будущих протоколов.

- **`images/`** — RGB-изображения.

## Демо

Запустите из корня проекта:

```bash
python scripts/bootstrap_demo_dataset.py
```

## Полный ISIC 2019

Скачайте данные с сайта челленджа / Kaggle (`ISIC_2019_Training_Input.zip`, `ISIC_2019_Training_GroundTruth.csv`, `ISIC_2019_Training_Metadata.csv`), распакуйте изображения в `images/` и соберите `ground_truth.csv` через готовый скрипт:

```bash
python scripts/prepare_isic2019.py \
    --gt        ISIC_2019_Training_GroundTruth.csv \
    --metadata  ISIC_2019_Training_Metadata.csv \
    --out       data/ground_truth.csv
```

Скрипт отбрасывает строки с `UNK ≥ 0.5`, переводит one-hot разметку в `label` и подтягивает `lesion_id` из метаданных.
