# Результаты и аудит

[← Главная](../README.md) · [Интерпретация валидации](../docs/VALIDATION.md)

| Файл | Что подтверждает |
|:--|:--|
| [`v58/public_result.json`](v58/public_result.json) | Сообщённый участником Public AP 0.97564 |
| [`v58/selection.json`](v58/selection.json) | Зафиксированный кандидат, точные метрики и ограничения |
| [`v58/selected_oof.csv`](v58/selected_oof.csv) | Все 198 OOF-прогнозов |
| [`v58/seed_confirmation.csv`](v58/seed_confirmation.csv) | Результаты трёх seed |
| [`v58/candidates.csv`](v58/candidates.csv) | 363 записи финального сравнения |
| [`v58/audit_summary.json`](v58/audit_summary.json) | Подсчёт исправленных/новых инверсий и изменений боксов |
| [`v58/original_49_train.csv`](v58/original_49_train.csv) | Исходные 49 проблемных train-боксов |
| [`v58/all_43_suspected_changes.csv`](v58/all_43_suspected_changes.csv) | 43 визуальные гипотезы на test, не GT |
| [`v58/spatial_checkpoint_replay.json`](v58/spatial_checkpoint_replay.json) | Повторное вычисление карт из шести DINO checkpoints |
| [`v52/all_experiments.csv`](v52/all_experiments.csv) | Исторические сравнения V52 |
| [`v52/original_replay_checks.json`](v52/original_replay_checks.json) | Проверка V52 в локальном пакете передачи |

Новые проверки именно текущей копии репозитория создаются командой `python scripts/reproduce.py` в `outputs/verification.json`. Они не заменяются сохранённым отчётом о предыдущем запуске.
