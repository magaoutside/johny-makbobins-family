"""Paired mixed-day blocked learning probe, separate from cross-day validation."""
from experiment import *
from sklearn.metrics import average_precision_score


def main():
    data = FeatureBank()
    rows = data.rows
    q, test, h = inputs()
    bo, _ = baseline(q, test)
    quality = np.zeros(len(rows))
    for index, r in q.iterrows():
        gt = h[h.frame_id.eq(r.frame_id) & h.class_name.eq(r.class_name)]
        z = iou(r[B], gt[B])
        quality[index] = z.max() if z.size else 0.
    names = ['anchor_d2', 'gt_frame_d3', 'geometry_quality_d2']
    ledger, predictions = [], []
    for day in DAYS:
        frames = sorted(rows.loc[rows.split.eq('oof') & rows.day.eq(day), 'frame_id'].unique())
        for block, held in enumerate(np.array_split(frames, 3)):
            val = rows.split.eq('oof') & rows.frame_id.isin(held)
            distance = np.abs(data.qt[:, None] - data.qt[None, val]).min(1)
            train = rows.split.eq('oof') & ~val & (distance > 120)
            # Withhold the whole temporal block from manually annotated prototypes too.
            lo, hi = data.qt[val].min() - 120, data.qt[val].max() + 120
            allowed = data.proto.loc[(data.pt < lo) | (data.pt > hi), 'frame_id']
            key = f'mixed_{day}_{block}'
            bank, groups = data.build(train, key, prototype_frames=allowed)
            for name in names:
                spec = specs()[name]
                columns = list(dict.fromkeys(c for g in spec['groups'] for c in groups[g])) + ['class_name']
                for seed in [23, 24, 25]:
                    cls = CatBoostRegressor if spec['target'] == 'quality' else CatBoostClassifier
                    model = cls(**model_parameters(spec, seed))
                    y = quality[train] if spec['target'] == 'quality' else rows.loc[train, 'error']
                    model.fit(bank.loc[train, columns], y, cat_features=['class_name'], sample_weight=np.ones(train.sum()))
                    p = probability(model, bank.loc[val, columns], spec)
                    path = A / f'models/probe_{name}_{day}_b{block}_s{seed}.cbm'
                    model.save_model(str(path))
                    predictions.append(rows.loc[val, ['id', 'day', 'class_name', 'error']].assign(
                        recipe=name, seed=seed, p_error=p, block=block))
                    ledger.append(dict(recipe=name, day=day, block=block, seed=seed,
                                       training_ids=rows.loc[train, 'id'].tolist(), held_ids=rows.loc[val, 'id'].tolist(),
                                       prototype_frames=sorted(set(allowed)), purge_seconds=120,
                                       features=columns, path=str(path.relative_to(A))))
            print('BLOCK', day, block, 'train', int(train.sum()), 'val', int(val.sum()), flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(A / 'features/refit_probe_predictions.csv', index=False)
    (A / 'models/refit_probe_manifest.json').write_text(json.dumps(ledger, indent=2))
    summary = []
    for name in names:
        mixed = predictions[predictions.recipe.eq(name)].groupby('id').p_error.mean().loc[q.id].to_numpy()
        old = pd.read_csv(A / f'features/{name}_initial_predictions.csv')
        opposite = old[old.split.eq('oof')].groupby('id').p_error.mean().loc[q.id].to_numpy()
        for label, p in [('opposite_day', opposite), ('mixed_days_blocked', mixed)]:
            s = bo.copy()
            person = q.class_name.eq('person')
            s[person] = (s[person] + .2 * p[person]) / 1.2
            summary.append(dict(recipe=name, validation=label, classifier_person_ap=average_precision_score(q.loc[person, 'error'], p[person]), **metrics(q, s)))
    pd.DataFrame(summary).to_csv(A / 'results/refit_probe.csv', index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
