"""Selected prototype features, six CatBoost models and the person correction.

The frozen upstream scores/features are explicit inputs. No historical folders,
feature search, test labels or per-ID overrides are used by this module.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .core import CLASSES, DAYS, aligned_csv, labels, queries, read_json, write_json


def timestamps(rows):
    return pd.to_datetime(rows.frame_id.str.rsplit('_', n=2).str[-2:].str.join(''),
                          format='%Y%m%d%H%M%S').astype('int64').to_numpy() / 1e9


def feature_bank(data_dir, fit_day, prefix='proto', purge_all=False):
    data_dir = Path(data_dir)
    rows = queries(data_dir)
    prototypes = pd.read_csv(data_dir / 'prototypes.csv')
    bank = aligned_csv(data_dir / 'static_features.csv', rows.id).drop(columns='id')
    if {'error', 'score'} & set(bank.columns):
        raise ValueError('Labels/scores must not enter model features')
    train = rows.split.eq('oof') & rows.day.eq(fit_day)
    qt, pt = timestamps(rows), timestamps(prototypes)
    for backbone in ['clip', 'vit_b', 'vit_l']:
        x = np.load(data_dir / 'embeddings' / f'{backbone}_queries.npy', allow_pickle=False)
        p = np.load(data_dir / 'embeddings' / f'{backbone}_prototypes.npy', allow_pickle=False)
        if len(x) != len(rows) or len(p) != len(prototypes):
            raise ValueError('Embedding row count mismatch')
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
        own = []
        for cls in CLASSES:
            use = prototypes.day.eq(fit_day) & prototypes.class_name.eq(cls)
            if not use.any():
                raise ValueError(f'No prototypes for {fit_day}, {cls}')
            similarity = x @ p[use].T
            valid = np.ones(similarity.shape, bool)
            purge = np.ones(len(rows), bool) if purge_all else train.to_numpy()
            valid[purge] = abs(qt[purge, None] - pt[None, use]) > 120
            top = np.sort(np.where(valid, similarity, -10), axis=1)[:, -min(3, use.sum()):]
            best = np.maximum(top, 0).sum(1) / np.maximum((top > -9).sum(1), 1)
            mean = np.where(valid, similarity, 0).sum(1) / np.maximum(valid.sum(1), 1)
            bank[f'{prefix}_{backbone}_{cls}_top3'] = best
            bank[f'{prefix}_{backbone}_{cls}_mean'] = mean
            own.append(best)
        own = np.stack(own, 1)
        class_index = rows.class_name.map(CLASSES.index).to_numpy()
        bank[f'{prefix}_{backbone}_margin'] = own[np.arange(len(rows)), class_index] - np.max(
            np.where(np.arange(4)[None, :] == class_index[:, None], -10, own), axis=1)
    return bank


def predict(data_dir, model_dir, config):
    rows = queries(data_dir)
    sums, counts = np.zeros(len(rows)), np.zeros(len(rows), dtype=int)
    for day in DAYS:
        train = rows.split.eq('oof') & rows.day.eq(day)
        evaluate = ~train
        bank = feature_bank(data_dir, day)
        for seed in config['seeds']:
            model = CatBoostClassifier()
            model.load_model(str(Path(model_dir) / f'person_day{day[-2:]}_seed{seed}.cbm'))
            if model.feature_names_ != config['features']:
                raise ValueError('Model feature names differ from config.json')
            sums[evaluate] += model.predict_proba(bank.loc[evaluate, config['features']])[:, 1]
            counts[evaluate] += 1
    if not (counts[rows.split.eq('oof')] == 3).all() or not (counts[rows.split.eq('test')] == 6).all():
        raise ValueError('Invalid cross-day ensemble coverage')
    previous = aligned_csv(Path(data_dir) / 'upstream_scores.csv', rows.id).dense_baseline.to_numpy()
    score = previous.copy()
    person = rows.class_name.eq('person').to_numpy()
    score[person] = (score[person] + 0.2 * (sums / counts)[person]) / 1.2
    return score


def train(data_dir, model_dir, config, threads=3):
    rows = queries(data_dir)
    y = labels(data_dir, rows)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ledger = []
    for day in DAYS:
        use = rows.split.eq('oof') & rows.day.eq(day)
        held = rows.split.eq('oof') & ~use
        if set(rows.loc[use, 'frame_id']) & set(rows.loc[held, 'frame_id']):
            raise ValueError('Training and validation frames overlap')
        bank = feature_bank(data_dir, day)
        for seed in config['seeds']:
            model = CatBoostClassifier(iterations=250, depth=2, learning_rate=.03,
                l2_leaf_reg=10, loss_function='Logloss', random_seed=seed,
                thread_count=threads, verbose=False, allow_writing_files=False)
            model.fit(bank.loc[use, config['features']], y[use].astype(int),
                      cat_features=['class_name'], sample_weight=np.ones(use.sum()))
            filename = f'person_day{day[-2:]}_seed{seed}.cbm'
            model.save_model(str(model_dir / filename))
            ledger.append({'file': filename, 'fit_day': day, 'seed': seed,
                'training_ids': rows.loc[use, 'id'].tolist(),
                'training_frames': sorted(rows.loc[use, 'frame_id'].unique()),
                'test_or_held_day_in_training': False})
            print(f'Trained {filename}: {use.sum()} training boxes', flush=True)
    write_json(model_dir / 'training.json', ledger)
    return ledger
