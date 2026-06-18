# Краткий анализ бейслайна (по семинарам)

Файл фиксирует, какие наработки каждого семинара вошли в итоговый бейслайн и в
какой артефакт они оформлены. Развёрнутый отчёт — `reports/final_report.md`,
сводный аудит — `reports/baseline_error_audit.md`.

## Семинар 1 — обучение базовой модели
ResNet18 (предобучен на ImageNet) + `CrossEntropyLoss` без весов классов,
`Adam(lr=1e-4)`, batch 32, 5 эпох. Код управляется конфигом
`configs/baseline.yaml` через `python -m src.train`. Итог: train_acc 97%,
val_acc 73%, классическое переобучение (val_loss растёт со 2-й эпохи). Кривые —
`runs/history.json`, график — `reports/figures/training_curves.png`.

## Семинар 2 — данные и честный сплит
Аудит данных и протокола (`src/data_audit.py` → `reports/data_audit.json`,
разбор — `reports/data_audit.md`): распределение классов, группировка по
`lesion_id`, реестр рисков R1–R9. Сплит train/val через `GroupShuffleSplit` по
`lesion_id` (`src/dataset.py`) исключает утечку кадров одного поражения
(пересечение групп = 0).

## Семинар 3 — стресс-тестирование
`src/transforms.py` задаёт восемь семейств искажений с сетками параметров и
«коридорами инвариантности» (`INVARIANT_RANGES`). `src/stress_test.py` строит
деградационные кривые (`reports/stress_metrics.csv`, `stress_summary.json`),
разбор — `reports/stress_test.md`. Самые болезненные искажения — шум и blur.

## Семинар 4 — решающее правило и рабочая точка
Формализация скора и решающего правила — `reports/score_and_decision_rule.md`.
Перебор порога с асимметричной функцией риска (`src/thresholds.py` →
`threshold_sweep.csv/json`, `operating_point.json`), выбор `τ* = 0.05` и его
обоснование — `reports/operating_point.md`. Анализ ошибок
(`reports/error_analysis.md` + картинки в `reports/error_cases/`), worst-slice
по подгруппам (`src/subgroups.py` → `slice_metrics.csv`,
`reports/worst_slice_analysis.md`) и калибровка (`src/calibration.py` →
`calibration.json`, reliability diagram).

## Что осознанно осталось вне бейслайна
- Веса классов / focal loss.
- Train-time аугментации, weight decay, ранняя остановка, LR-scheduler.
- Temperature scaling после расчёта ECE.
- Stratified group k-fold для доверительных интервалов.
- Использование метаданных пациента.

Эти пункты — в разделе «Направления развития» итогового отчёта.
