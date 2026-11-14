"""Reproduce V19 ranking from saved V18 scores and cross-day V19 features.

No training, image downloads, test labels, or earlier project directories are
needed for this replay. Feature extraction scripts and their provenance are
saved separately; replay is not end-to-end retraining of V18.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def predict(data, body_weight=.75, hand_weight=.5):
    base = data.baseline_v18.to_numpy(float)
    body = (
        data.class_name.eq('person').to_numpy()
        & data.body_point_eligible.eq(1).to_numpy()
        & (data.body_height_ratio.to_numpy() < .85)
        & (data.body_top_error.to_numpy() < .2)
        & data.body_width_ratio.between(.65, 1.5).to_numpy()
        & data.seat_back.eq(1).to_numpy()
        & data.seat_head_count.eq(1).to_numpy()
        & (data.seat_template_n.to_numpy() >= 3)
        & (data.seat_distance.to_numpy() < 1.2)
        & (data.y2.to_numpy() < 710)
        & (data.x1.to_numpy() > 5)
        & (data.x2.to_numpy() < 1275)
        & (data.body_error_probability.to_numpy() >= .5)
        & (data.head_coverage.to_numpy() >= .9)
    )
    hand = (
        data.class_name.eq('nude_hands').to_numpy()
        & data.mp_assigned.eq(1).to_numpy()
        & (data.mp_iou.to_numpy() >= .5)
        & (data.mp_views.to_numpy() >= 4)
        & (data.mp_turns.to_numpy() >= 3)
        & (data.mp_coherence.to_numpy() >= .65)
        & (data.mp_point_fraction.to_numpy() >= .85)
        & (data.mp_palm_fraction.to_numpy() >= .8)
        & (data.mp_instances.to_numpy() <= 1)
        & ~((data.yolo_old_instances.to_numpy() >= 2)
            & (data.yolo_refined_instances.to_numpy() >= 2))
        & data.duplicate_veto.eq(0).to_numpy()
        & (data.wevis_hand_instances.to_numpy() < 2)
    )
    assert not (body & hand).any()
    candidate = (data.baseline_v16.to_numpy()
                 + body_weight * (1 - data.baseline_v16.to_numpy())
                 * data.body_error_probability.to_numpy())
    score = np.where(body, np.maximum(base, candidate), base)
    target = np.clip((.8 - data.mp_iou.to_numpy()) / .6, 0, 1)
    score -= hand_weight * np.maximum(base - target, 0) * hand
    assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['main', 'conservative'], default='main')
    parser.add_argument('--split', choices=['test', 'oof'], default='test')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / 'replay_inputs' / f'{args.split}.csv')
    weight = .75 if args.variant == 'main' else .5
    score = predict(data, body_weight=weight)
    expected = pd.read_csv(root / ('submission_v19_' + args.variant + '.csv')) if args.split == 'test' else pd.read_csv(root / 'release_oof' / (args.variant + '.csv'))
    assert data.id.tolist() == expected.id.tolist()
    np.testing.assert_allclose(score, expected.score, atol=1e-12, rtol=0)
    path = args.output or root / f'replayed_{args.split}_{args.variant}.csv'
    data[['id']].assign(score=score).to_csv(path, index=False)
    print(f'Exact numerical replay verified: {len(score)} rows -> {path}')
    if args.split == 'oof':
        from sklearn.metrics import average_precision_score
        print('Pooled AP:', average_precision_score(data.error, score))
        for day, group in data.groupby('day'):
            print(day, average_precision_score(group.error, score[group.index]))


if __name__ == '__main__':
    main()
