"""Selected TabICL experts: opposite-day fitting, portable loading and prior shift."""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from .core import DAYS, CLASSES, queries, labels, aligned_csv, write_json
from .prototype import feature_bank

SEEDS = (5601, 5602, 5603)
FAMILIES = ('anchor', 'full', 'crop')


def bank(data, rows, day, family):
    x = feature_bank(data, day, prefix='gt', purge_all=True)
    # Original TabICL bank places the categorical column after all numeric ones.
    x = x[[c for c in x if c != 'class_name'] + ['class_name']]
    if family != 'anchor':
        extra = aligned_csv(Path(data) / f'sam2_{family}.csv', rows.id).drop(columns='id')
        x = pd.concat([x, extra], axis=1).fillna(0)
    if {'id', 'score', 'error', 'frame_id', 'day'} & set(x):
        raise ValueError('Forbidden target/identity feature')
    return x


def filename(family, day, seed):
    return f'tabular_{family}_day{day[-2:]}_seed{seed}.pkl'


@contextmanager
def relocate(checkpoint, device):
    # TabICL 2.2.0 unpickles before applying load(device=...). Patch only inside
    # this scope so an old absolute model path cannot be resolved accidentally.
    from tabicl._sklearn.base import TabICLBaseEstimator
    original = TabICLBaseEstimator.__setstate__
    def restore(self, state):
        state = dict(state)
        state.update(model_path=str(Path(checkpoint).resolve()), device=device, n_jobs=2)
        original(self, state)
    TabICLBaseEstimator.__setstate__ = restore
    try:
        yield
    finally:
        TabICLBaseEstimator.__setstate__ = original


def run(data, models, checkpoint, device, fit=False):
    import torch
    from tabicl import TabICLClassifier
    torch.set_num_threads(2)
    rows = queries(data)
    y = labels(data, rows) if fit else None
    models = Path(models)
    if fit:
        models.mkdir(parents=True, exist_ok=True)
    predictions, ledger = [], []
    for family in FAMILIES:
        for day in DAYS:
            tr = rows.split.eq('oof') & rows.day.eq(day)
            ev = ~tr
            x = bank(data, rows, day, family)
            if not set(rows.loc[tr, 'frame_id']).isdisjoint(rows.loc[ev, 'frame_id']):
                raise ValueError('Train/evaluation frame overlap')
            for seed in SEEDS:
                path = models / filename(family, day, seed)
                if fit:
                    model = TabICLClassifier(
                        n_estimators=8 if family == 'anchor' and seed == 5601 else 16,
                        model_path=str(checkpoint), allow_auto_download=False,
                        device=device, n_jobs=2, batch_size=2, random_state=seed,
                        use_amp=False, verbose=False)
                    model.fit(x.loc[tr], y.loc[tr].astype(int))
                    model.save(str(path), save_model_weights=False,
                               save_training_data=True, save_kv_cache=False)
                    ledger.append(dict(file=path.name, fit_day=day, seed=seed,
                                       training_ids=rows.loc[tr, 'id'].tolist(),
                                       features=x.columns.tolist()))
                else:
                    with relocate(checkpoint, device):
                        model = TabICLClassifier.load(str(path), device=device)
                prob = model.predict_proba(x.loc[ev])[:, 1]
                predictions.append(rows.loc[ev, ['id', 'split', 'day', 'class_name']].assign(
                    fit_day=day, family=family, seed=seed, probability=prob))
                print(f'TabICL: {family}, fit {day}, seed {seed}', flush=True)
                del model
    if fit:
        write_json(models / 'tabular_training.json', ledger)
        write_json(models / 'source_priors.json', source_priors(data))
    return pd.concat(predictions, ignore_index=True)


def source_priors(data):
    rows = queries(data)
    y = labels(data, rows)
    return {day: {c: float((y[m].sum() + 1) / (m.sum() + 2))
                  for c in CLASSES
                  for m in [rows.split.eq('oof') & rows.day.eq(day) & rows.class_name.eq(c)]}
            for day in DAYS}


def adapt_prior(p, prior):
    p = np.clip(np.asarray(p, float), 1e-4, 1-1e-4)
    logit = lambda v: np.log(v / (1-v))
    target = prior
    for _ in range(100):
        posterior = 1 / (1 + np.exp(-(logit(p) + logit(target) - logit(prior))))
        new = (posterior.sum() + 10 * prior) / (len(p) + 10)
        if abs(new-target) < 1e-8:
            break
        target = new
    return 1 / (1 + np.exp(-(logit(p) + logit(target) - logit(prior))))


def aggregate(predictions, rows, priors):
    group = ['id', 'split', 'day', 'class_name', 'fit_day', 'family']
    d = predictions.groupby(group, as_index=False).probability.mean()
    anchor = d.family.eq('anchor')
    for keys, g in d[anchor].groupby(['fit_day', 'split', 'day', 'class_name']):
        fit_day, _, _, cls = keys
        d.loc[g.index, 'probability'] = adapt_prior(g.probability, priors[fit_day][cls])
    return {family: d[d.family.eq(family)].groupby('id').probability.mean().loc[rows.id].to_numpy()
            for family in FAMILIES}
