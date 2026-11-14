# V58 · Public AP 0.97564

Финальная формула объединяет V52, SAM2/TabICL-поправки V57, DINO-B spatial с RGB flip TTA для `cup` и калибровку `tray`. Максимальное отклонение от V52 — 0.15.

```bash
# Из корня репозитория
python scripts/reproduce.py --version v58
```

- `replay.py` — читаемая финальная формула.
- `replay_inputs.npz` — замороженные входы экспертов для 198 OOF + 362 test query.
- `models/` — шесть DINO-B spatial голов и metadata, seeds 3101/3102/3103.
- `precursor_models/` — сохранённые SAM2/TabICL и anchor TabICL модели, использованные в предыдущей части рецепта.

Быстрый replay собирает score из готовых экспертных прогнозов. Веса голов включены для сохранности и RGB-ветки; при этом replay не делает нейросетевой инференс. Для загрузки `.pkl` нужен исходный TabICL runtime и отдельный foundation checkpoint.

[Метод](../../docs/METHOD.md) · [Полный код экспериментов](../../research/rascar_v58/) · [Оригинальный сабмит](../../submissions/v58.csv)
