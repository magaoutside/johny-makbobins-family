"""Train-only OWLv2 boundary convention calibration with purged inner features."""
from experiment import *


def main():
    data = FeatureBank()
    rows = data.rows
    manual = pd.read_csv(ROOT / 'rascar_v51/inputs/manual_records.csv').set_index('id')
    observations = []
    for line in (ROOT / 'rascar_v51/features/manual.jsonl').read_text().splitlines():
        record = json.loads(line)
        gt = manual.loc[record['id'], B].to_numpy(float)
        refs = [r for r in record['references'] if r['positive'] >= .1]
        if not refs:
            continue
        boxes = np.array([r['xyxy'] for r in refs])
        overlaps = iou(gt, boxes)[0]
        j = overlaps.argmax()
        if overlaps[j] < .35:
            continue
        reference = boxes[j]
        offsets = np.clip((gt - reference) / np.tile(np.maximum(reference[2:] - reference[:2], 1), 2), -.4, .4)
        observations.append(dict(id=record['id'], frame_id=record['frame_id'], day=record['day'],
                                 class_name=record['class_name'], **dict(zip(['left', 'top', 'right', 'bottom'], offsets))))
    observations = pd.DataFrame(observations)
    observations.to_csv(A / 'inputs/owl_calibration_observations.csv', index=False)
    source_time = timestamps(observations)
    records = {}
    for split in ['oof', 'test']:
        for line in (ROOT / f'rascar_v50/features/{split}.jsonl').read_text().splitlines():
            record = json.loads(line)
            assert record['id'] not in records
            records[record['id']] = record
    assert set(records) == set(rows.id)
    out, ledger = [], []
    for fit in DAYS + ['full']:
        for index, query in rows.iterrows():
            fit_days = DAYS if fit == 'full' else [fit]
            use = observations.day.isin(fit_days) & observations.class_name.eq(query.class_name)
            if query.split == 'oof' and query.day in fit_days:
                use &= abs(source_time - data.qt[index]) > 120
            values = observations.loc[use, ['left', 'top', 'right', 'bottom']].to_numpy()
            correction = np.median(values, axis=0) * len(values) / (len(values) + 10) if len(values) else np.zeros(4)
            row = dict(id=query.id, fit_day=fit, calibration_count=len(values))
            record = records[query.id]
            box = query[B].to_numpy(float)
            area = np.prod(np.maximum(box[2:] - box[:2], 1))
            for threshold in [.05, .1, .2]:
                refs = [r for r in record['references'] if r['positive'] >= threshold]
                tag = f'c{round(threshold * 100):02}'
                old = np.array([r['xyxy'] for r in refs]).reshape(-1, 4)
                adjusted = old + correction * np.tile(np.maximum(old[:, 2:] - old[:, :2], 1), (1, 2))
                for label, boxes in [('old', old), ('new', adjusted)]:
                    if len(boxes):
                        overlaps = iou(box, boxes)[0]
                        reference = boxes[overlaps.argmax()]
                        intersection = np.prod(np.maximum(0, np.minimum(box[2:], reference[2:]) - np.maximum(box[:2], reference[:2])))
                        values = dict(iou=float(overlaps.max()), cover_query=intersection / area,
                                      cover_ref=intersection / max(np.prod(reference[2:] - reference[:2]), 1), missing=0)
                    else:
                        values = dict(iou=0., cover_query=0., cover_ref=0., missing=1)
                    row.update({f'{tag}_{label}_{name}': value for name, value in values.items()})
                row[f'{tag}_iou_change'] = row[f'{tag}_new_iou'] - row[f'{tag}_old_iou']
            out.append(row)
        for cls in CL:
            use = observations.day.isin(DAYS if fit == 'full' else [fit]) & observations.class_name.eq(cls)
            values = observations.loc[use, ['left', 'top', 'right', 'bottom']].to_numpy()
            delta = np.median(values, axis=0) * len(values) / (len(values) + 10) if len(values) else np.zeros(4)
            ledger.append(dict(fit_day=fit, class_name=cls, n=len(values),
                               correction=delta.tolist(), manual_ids=observations.loc[use, 'id'].tolist()))
    pd.DataFrame(out).to_csv(A / 'features/calibrated_owl.csv', index=False)
    (A / 'models/owl_calibration.json').write_text(json.dumps(ledger, indent=2))
    print(pd.DataFrame(ledger).drop(columns='manual_ids').to_string(index=False))


if __name__ == '__main__':
    main()
