# Исследовательский код

[← Главная](../README.md) · [Метод](../docs/METHOD.md) · [Запуск готовых версий](../docs/REPRODUCIBILITY.md)

Здесь **600 файлов доступных авторских исходников**: все этапы до V58, включая неудачные эксперименты, вспомогательные зависимости и версии Kaggle. Имена каталогов сохранены, чтобы можно было проследить происхождение признаков и моделей.

## Начать с финальных версий

| Версия / этап | Входные файлы |
|:--|:--|
| V58 — выбранная визуальная модель | [`gpu_dense.py`](rascar_v58/gpu_dense.py), [`gpu_spatial_tta.py`](rascar_v58/gpu_spatial_tta.py) |
| V58 — остальные GPU-эксперименты | [`gpu_experts.py`](rascar_v58/gpu_experts.py), [`gpu_jackknife.py`](rascar_v58/gpu_jackknife.py), [`gpu_finetune.py`](rascar_v58/gpu_finetune.py) |
| V58 — выбор, устойчивость, выпуск | [`study.py`](rascar_v58/study.py), [`stress_test.py`](rascar_v58/stress_test.py), [`finalize.py`](rascar_v58/finalize.py) |
| V57 — коррекция выбросов | [`correct_outliers.py`](rascar_v56/correct_outliers.py) |
| V56 — GPU общий код | [`kaggle_payload/rascar_v56/`](rascar_v56/kaggle_payload/rascar_v56/) |
| V52 — подготовка и обучение | [`prepare.py`](rascar_v52/prepare.py), [`train.py`](rascar_v52/train.py) |
| V52 — выпуск | [`finalize.py`](rascar_v52/finalize.py), [`replay_candidate.py`](rascar_v52/replay_candidate.py) |
| Базовые признаки и dense | [`rascar_v31/`](rascar_v31/), [`rascar_v41/`](rascar_v41/) |

## Как читать архив

`SOURCE_INDEX.csv` содержит пути и SHA256. `.py` и `.sh` сохранены без изменений. У `.ipynb` сохранены исходники ячеек, очищены outputs и execution counts.

Исторические скрипты используют исходный layout рабочего проекта и иногда абсолютные пути `/kaggle/working`. Им нужны промежуточные данные. Запускать всю историю последовательным `python *.py` не следует: среди файлов есть отдельные эксперименты, составители отчётов и альтернативные ветки.

Часть исторических строк описывает состояние до Public-отправки либо называет записи таблиц «независимыми экспериментами». Актуальная интерпретация, статус и ограничения приведены в корневом README и `docs/`; исторический код не переписывался задним числом.

Для проверенного запуска двух финальных сабмитов используйте:

```bash
python scripts/reproduce.py --version all
```

Для обучения с нуля начните с [инструкции по данным и GPU](../docs/REPRODUCIBILITY.md#обучение-и-rgb-инференс).
