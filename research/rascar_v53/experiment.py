"""V53: purged visual exemplars, small-data controls, and full-data refits.

No held-day labels enter features, preprocessing, or model fitting. All comparisons
remain development experiments, including the historically selected V52 baseline.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
A = Path(__file__).resolve().parent
V = ROOT / 'rascar_v52'
sys.path.insert(0, str(ROOT / 'rascar_v41'))
from common import inputs, baseline, metrics, DAYS, CL, B, iou


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def timestamps(frame):
    return pd.to_datetime(frame.frame_id.str.rsplit('_', n=2).str[-2:].str.join(''),
                          format='%Y%m%d%H%M%S').astype('int64').to_numpy() / 1e9


class FeatureBank:
    def __init__(self):
        self.rows = pd.read_csv(V / 'inputs/rows.csv')
        self.static = pd.read_csv(V / 'features/static.csv').set_index('id').loc[self.rows.id].reset_index(drop=True)
        self.groups = json.loads((V / 'inputs/groups.json').read_text())
        self.proto = pd.read_csv(V / 'inputs/prototypes.csv')
        self.qt, self.pt = timestamps(self.rows), timestamps(self.proto)
        self.emb = {e: np.load(V / f'features/{e}.npy') for e in
                    ['clip', 'vit_b', 'vit_l', 'siglip2', 'dino_small', 'owl']}
        self.pe = {e: np.load(V / f'features/{e}_prototypes.npy') for e in ['clip', 'vit_b', 'vit_l']}
        self.cache = {}

    def build(self, train, key, prototype_frames=None):
        train = np.asarray(train, bool)
        assert not self.rows.loc[train, 'split'].eq('test').any()
        if key in self.cache:
            return self.cache[key]
        rows = self.rows
        frame_ids = set(rows.loc[train, 'frame_id'])
        # The manual GT bank also has three frames with no official error query.
        # Preserve them for cross-day/full fits, exactly as the V52 incumbent did.
        if key in DAYS or key == 'full':
            fit_days = set(rows.loc[train, 'day'])
            frame_ids = set(self.proto.loc[self.proto.day.isin(fit_days), 'frame_id'])
        if prototype_frames is not None:
            frame_ids = set(prototype_frames)
        columns = {}
        groups = {k: list(v) for k, v in self.groups.items()}
        for view, start in [('roi', 1), ('context', 0), ('contrast', 2)]:
            group = 'gt' if view == 'roi' else 'gt_' + view
            groups[group] = []
            for e in ['clip', 'vit_b', 'vit_l']:
                width = self.emb[e].shape[1] // 3
                x = normalize(self.emb[e][:, start * width:(start + 1) * width])
                p = normalize(self.pe[e][:, start * width:(start + 1) * width])
                own = []
                for cls in CL:
                    use = self.proto.frame_id.isin(frame_ids) & self.proto.class_name.eq(cls)
                    assert use.any(), (key, cls)
                    sim = x @ p[use].T
                    valid = abs(self.qt[:, None] - self.pt[None, use]) > 120
                    top = np.sort(np.where(valid, sim, -10), axis=1)[:, -min(3, use.sum()):]
                    best = np.maximum(top, 0).sum(1) / np.maximum((top > -9).sum(1), 1)
                    mean = np.where(valid, sim, 0).sum(1) / np.maximum(valid.sum(1), 1)
                    for name, values in [('top3', best), ('mean', mean)]:
                        label = f'{group}_{e}_{cls}_{name}'
                        columns[label] = values
                        groups[group].append(label)
                    own.append(best)
                own = np.stack(own, 1)
                ci = rows.class_name.map(CL.index).to_numpy()
                label = f'{group}_{e}_margin'
                columns[label] = own[np.arange(len(rows)), ci] - np.max(
                    np.where(np.arange(4)[None, :] == ci[:, None], -10, own), axis=1)
                groups[group].append(label)
        # Top matches from DISTINCT source frames, to avoid repeated objects dominating.
        groups['gt_frame'] = []
        for e in ['clip', 'vit_b', 'vit_l']:
            width = self.emb[e].shape[1] // 3
            x = normalize(self.emb[e][:, width:2 * width])
            p = normalize(self.pe[e][:, width:2 * width])
            for cls in CL:
                use = self.proto.frame_id.isin(frame_ids) & self.proto.class_name.eq(cls)
                sim = x @ p[use].T
                valid = abs(self.qt[:, None] - self.pt[None, use]) > 120
                source_frames = self.proto.loc[use, 'frame_id'].to_numpy()
                unique_frames = sorted(set(source_frames))
                per_frame = np.stack([np.where(valid, sim, -10)[:, source_frames == f].max(1)
                                      for f in unique_frames], axis=1)
                top = np.sort(per_frame, axis=1)[:, -min(3, len(unique_frames)):]
                label = f'gtframe_{e}_{cls}_top3'
                columns[label] = np.maximum(top, 0).sum(1) / np.maximum((top > -9).sum(1), 1)
                groups['gt_frame'].append(label)
        groups['supervised_visual'] = []
        groups['supervised_geometry'] = []
        sources = {}
        for e, raw in self.emb.items():
            width = raw.shape[1] // 3
            sources[e + '_roi'] = normalize(raw[:, width:2 * width])
            if e in ['clip', 'vit_l']:
                sources[e + '_contrast'] = normalize(raw[:, 2 * width:])
        geometry = self.static[groups['base23']].to_numpy()
        scale = np.maximum(geometry[train].std(0), .1)
        z = (geometry - geometry[train].mean(0)) / scale
        sources['geometry'] = z
        eligible = train[None, :] & rows.class_name.to_numpy()[:, None].__eq__(rows.class_name.to_numpy()[None, :])
        eligible &= abs(self.qt[:, None] - self.qt[None, :]) > 120
        y = rows.error.fillna(0).to_numpy()
        for source, x in sources.items():
            sim = -np.mean((x[:, None] - x[None, :]) ** 2, axis=2) if source == 'geometry' else x @ x.T
            group = 'supervised_geometry' if source == 'geometry' else 'supervised_visual'
            outputs = []
            for target in [0, 1]:
                valid = eligible & (y[None, :] == target)
                ranked = np.sort(np.where(valid, sim, -1e8), axis=1)
                count = valid.sum(1)
                best = np.where(count > 0, ranked[:, -1], 0)
                top = ranked[:, -3:]
                average = np.where(top > -1e7, top, 0).sum(1) / np.maximum((top > -1e7).sum(1), 1)
                outputs.append(average)
                for name, values in [('top1', best), ('top3', average), ('missing', (count == 0).astype(float))]:
                    label = f'sup_{source}_y{target}_{name}'
                    columns[label] = values
                    groups[group].append(label)
            label = f'sup_{source}_error_minus_correct'
            columns[label] = outputs[1] - outputs[0]
            groups[group].append(label)
            temperature = 1. if source == 'geometry' else .05
            logits = np.where(eligible, sim / temperature, -1e8)
            weights = np.exp(np.clip(logits - logits.max(1, keepdims=True), -80, 0)) * eligible
            label = f'sup_{source}_neighbor_error'
            columns[label] = (weights @ y + .5) / (weights.sum(1) + 1)
            groups[group].append(label)
        # Geometry interactions are deterministic and contain no GT.
        groups['geometry_derived'] = []
        for side in ['top', 'bottom', 'left', 'right']:
            v = self.static[side + '_offset'].to_numpy()
            for kind, value in [('abs', abs(v)), ('positive', np.maximum(v, 0)), ('negative', np.maximum(-v, 0))]:
                label = f'offset_{side}_{kind}'
                columns[label] = value
                groups['geometry_derived'].append(label)
        for label, value in {
            'cover_min': self.static[['cover_query', 'cover_ref']].min(axis=1),
            'cover_gap': abs(self.static.cover_query - self.static.cover_ref),
            'sam_wevis_gap': abs(self.static.sam_score - self.static.wevis_score),
            'pose_geometry_gap': self.static.pose_best_iou - self.static.best_iou,
        }.items():
            columns[label] = value
            groups['geometry_derived'].append(label)
        bank = pd.concat([self.static, pd.DataFrame(columns)], axis=1)
        if (A / 'features/calibrated_owl.csv').exists() and key in DAYS + ['full']:
            calibrated = pd.read_csv(A / 'features/calibrated_owl.csv')
            calibrated = calibrated[calibrated.fit_day.eq(key)].set_index('id').loc[rows.id]
            names = [c for c in calibrated if c not in ['fit_day']]
            groups['calibrated'] = ['calibrated_' + c for c in names]
            bank = pd.concat([bank, calibrated[names].reset_index(drop=True).set_axis(groups['calibrated'], axis=1)], axis=1)
        assert np.isfinite(bank.drop(columns='class_name').to_numpy()).all()
        self.cache[key] = (bank, groups)
        return bank, groups


def specs():
    families = {
        'anchor': ['base23', 'gt'],
        'gt_frame': ['base23', 'gt_frame'],
        'gt_context': ['base23', 'gt', 'gt_context'],
        'gt_contrast': ['base23', 'gt', 'gt_contrast'],
        'gt_allviews': ['base23', 'gt', 'gt_context', 'gt_contrast', 'gt_frame'],
        'labeled_visual': ['base23', 'supervised_visual'],
        'labeled_geometry': ['base23', 'supervised_geometry'],
        'labeled_all': ['base23', 'gt', 'supervised_visual', 'supervised_geometry'],
        'geometry': ['base23', 'gt', 'geometry_derived'],
        'owl_text': ['base23', 'gt', 'owl', 'clip_text'],
        'frozen_all': ['frozen_all', 'gt'],
    }
    configurations = {}
    for family, groups in families.items():
        for depth in [2, 3]:
            configurations[f'{family}_d{depth}'] = dict(groups=groups, depth=depth, mode='all', target='error')
    for depth in [1, 2, 3]:
        configurations[f'anchor_person_d{depth}'] = dict(groups=families['anchor'], depth=depth, mode='person', target='error')
    for mode in ['ordered', 'person_weight', 'no_bootstrap', 'regularized']:
        configurations[f'anchor_{mode}_d2'] = dict(groups=families['anchor'], depth=2, mode=mode, target='error')
    for family in ['anchor', 'labeled_all', 'geometry']:
        configurations[f'{family}_quality_d2'] = dict(groups=families[family], depth=2, mode='all', target='quality')
    if (A / 'features/calibrated_owl.csv').exists():
        for depth in [2, 3]:
            configurations[f'calibrated_d{depth}'] = dict(groups=['base23', 'gt', 'calibrated'], depth=depth, mode='all', target='error')
    return configurations


def model_parameters(spec, seed):
    params = dict(iterations=250, depth=spec['depth'], learning_rate=.03, l2_leaf_reg=10,
                  random_seed=seed, thread_count=2, verbose=False, allow_writing_files=False)
    if spec['mode'] == 'ordered':
        params['boosting_type'] = 'Ordered'
    if spec['mode'] == 'no_bootstrap':
        params['bootstrap_type'] = 'No'
    if spec['mode'] == 'regularized':
        params.update(iterations=400, l2_leaf_reg=30, random_strength=2)
    return params


def probability(model, X, spec):
    if spec['target'] == 'quality':
        # Fixed transform centered at official IoU threshold; not a calibrated probability.
        return expit((.6 - np.clip(model.predict(X), 0, 1)) / .12)
    return model.predict_proba(X)[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', nargs='*')
    parser.add_argument('--seeds', nargs='+', type=int, default=[23, 24, 25])
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--tag', default='initial')
    args = parser.parse_args()
    data = FeatureBank()
    rows = data.rows
    q, te, h = inputs()
    bo, bt = baseline(q, te)
    incumbent = pd.read_csv(V / 'results/oof/purged_ensemble_d2_person_a20.csv').set_index('id').loc[q.id, 'score'].to_numpy()
    quality = np.zeros(len(rows))
    for i, r in q.iterrows():
        gt = h[h.frame_id.eq(r.frame_id) & h.class_name.eq(r.class_name)]
        overlaps = iou(r[B], gt[B])
        quality[i] = overlaps.max() if overlaps.size else 0.
    assert rows.loc[rows.split.eq('oof'), 'id'].tolist() == q.id.tolist()
    fits = DAYS + (['full'] if args.full else [])
    banks = {}
    for fit in fits:
        train = rows.split.eq('oof') & (True if fit == 'full' else rows.day.eq(fit))
        banks[fit] = data.build(train, fit)
    manifest = []
    scores = []
    started = time.time()
    for name, spec in specs().items():
        if args.names and name not in args.names:
            continue
        preds = []
        for fit in fits:
            bank, groups = banks[fit]
            columns = list(dict.fromkeys(c for g in spec['groups'] for c in groups[g])) + ['class_name']
            assert not set(columns) & {'id', 'frame_id', 'day', 'score', 'error'}
            train = rows.split.eq('oof') & (True if fit == 'full' else rows.day.eq(fit))
            if spec['mode'] == 'person':
                train &= rows.class_name.eq('person')
            evaluate = rows.split.eq('test') if fit == 'full' else (rows.split.eq('test') | (rows.split.eq('oof') & rows.day.ne(fit)))
            y = quality[train] if spec['target'] == 'quality' else rows.loc[train, 'error']
            weight = np.where(rows.loc[train, 'class_name'].eq('person'), 3., 1.) if spec['mode'] == 'person_weight' else np.ones(int(train.sum()))
            for seed in args.seeds:
                path = A / f'models/{name}_s{seed}_fit{fit[-2:]}.cbm'
                cls = CatBoostRegressor if spec['target'] == 'quality' else CatBoostClassifier
                model = cls(**model_parameters(spec, seed))
                model.fit(bank.loc[train, columns], y, cat_features=['class_name'], sample_weight=weight)
                model.save_model(str(path))
                p = probability(model, bank.loc[evaluate, columns], spec)
                preds.append(rows.loc[evaluate, ['id', 'split', 'day', 'class_name']].assign(p_error=p, fit_day=fit, seed=seed))
                manifest.append(dict(name=name, spec=spec, fit_day=fit, seed=seed, features=columns,
                                     model_path=str(path.relative_to(A)), train_ids=rows.loc[train, 'id'].tolist(),
                                     prototype_frames=sorted(data.proto.loc[data.proto.day.isin(DAYS if fit == 'full' else [fit]), 'frame_id'].unique()),
                                     supervised_prototype_training_only=True, purge_seconds=120,
                                     parameters=model_parameters(spec, seed)))
        pred = pd.concat(preds, ignore_index=True)
        pred.to_csv(A / f'features/{name}_{args.tag}_predictions.csv', index=False)
        oo = pred[pred.split.eq('oof')].groupby('id').p_error.mean().loc[q.id].to_numpy()
        tt = pred[pred.split.eq('test') & pred.fit_day.ne('full')].groupby('id').p_error.mean().loc[te.id].to_numpy()
        for alpha in [.1, .2, .3, .4]:
            s, z = bo.copy(), bt.copy()
            mask, mt = q.class_name.eq('person'), te.class_name.eq('person')
            s[mask] = (bo[mask] + alpha * oo[mask]) / (1 + alpha)
            z[mt] = (bt[mt] + alpha * tt[mt]) / (1 + alpha)
            exp = f'{name}_{args.tag}_person_a{round(alpha * 100):02}'
            q.assign(score=s).to_csv(A / f'results/oof/{exp}.csv', index=False)
            te[['id']].assign(score=z).to_csv(A / f'results/submissions/{exp}.csv', index=False)
            scores.append(dict(name=exp, recipe=name, alpha=alpha, **metrics(q, s),
                               delta_v52=metrics(q, s)['ap'] - metrics(q, incumbent)['ap']))
        pd.DataFrame(scores).to_csv(A / f'results/experiments_{args.tag}.csv', index=False)
        (A / f'models/manifest_{args.tag}.json').write_text(json.dumps(manifest, indent=2))
        print(name, 'best', max(x['ap'] for x in scores[-4:]), 'seconds', round(time.time() - started, 1), flush=True)
    print(pd.DataFrame(scores).sort_values('ap', ascending=False).head(15).to_string(index=False))


if __name__ == '__main__':
    main()
