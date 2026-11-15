"""Regenerate training-pool checks and pseudo labels from competition JPG/CSV.

Ports the selected pool_teacher_v2 and pool_review steps. Model scores are
computed afresh; the separate review ledger contains only annotation decisions.
YOLO and DINOv3 run in their respective environments, without Kaggle-only paths.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from rascar.core import ROOT, BBOX as B, DAYS
from rascar.sam3 import table

CLASSES = ['person', 'cup', 'nude_hands', 'tray']
DECISIONS = ['id', 'day', 'review_index', 'visual_decision', 'visual_reason', 'review_source']


def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    inter = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                       - np.maximum(a[:, None, :2], b[None, :, :2])).prod(2)
    area_a, area_b = (a[:, 2:] - a[:, :2]).prod(1), (b[:, 2:] - b[:, :2]).prod(1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def pool_metadata(raw, out):
    """Original >600-second exclusion, using only timestamps and frame IDs."""
    frames, machine = table(raw, 'machine_frames.csv'), table(raw, 'machine_predictions.csv')
    manual = table(raw, 'human_boxes_train.csv')
    train_frames, test = table(raw, 'train_frames.csv'), table(raw, 'test.csv')
    train_frames = train_frames[train_frames.frame_id.isin(manual.frame_id)]
    used = set(manual.frame_id) | set(test.frame_id)
    stamps = pd.to_datetime(frames.timestamp, utc=True, format='mixed')
    gold = pd.to_datetime(train_frames.timestamp, utc=True, format='mixed').astype('int64').to_numpy()
    frames['day'] = stamps.dt.strftime('%Y-%m-%d')
    frames['nearest_annotated_seconds'] = np.abs(stamps.astype('int64').to_numpy()[:, None] - gold).min(1) / 1e9
    frames['eligible_time'] = frames.nearest_annotated_seconds.gt(600) & ~frames.frame_id.isin(used) & frames.day.isin(DAYS)
    frames['hour'] = stamps.dt.hour
    frames['scored_proposals'] = frames.frame_id.map(machine[machine.class_name.ne('table')].groupby('frame_id').size()).fillna(0).astype(int)
    out.mkdir(parents=True, exist_ok=True)
    frames.to_csv(out / 'pool_frame_ledger.csv', index=False)
    for day in DAYS:
        selected = frames[frames.eligible_time & frames.day.eq(day)]
        rows = machine[machine.frame_id.isin(selected.frame_id) & machine.class_name.isin(CLASSES)].copy()
        rows['id'], rows['day'], rows['source'] = rows.detection_id, day, 'machine_pool'
        if rows.id.duplicated().any():
            raise ValueError('Machine detection IDs must be unique')
        folder = out / day
        folder.mkdir(exist_ok=True)
        rows.to_csv(folder / 'candidates.csv', index=False)
    return frames


def checkpoint(args, kind, day):
    if args.work is not None:
        paths = {'generic': f'local_evidence/weights/{day}.pt',
                 'hand': f'modes_and_sahi/weights/hand_{day}.pt',
                 'semantic': f'geometry_guards/semantic_both_{day}.joblib'}
        path = args.work / paths[kind]
    else:
        names = {'generic': f'generic_{day}.pt', 'hand': f'hand_initial_{day}.pt',
                 'semantic': f'pool_semantic_{day}.joblib'}
        path = args.models / names[kind]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def image_reader(raw):
    from PIL import Image
    paths = {p.stem: p for p in raw.rglob('*.jpg')}

    def read(fid):
        with Image.open(paths[fid]) as im:
            return im.convert('RGB').resize((1280, 720), Image.Resampling.BICUBIC)
    return paths, read


def hand_checks(rows, model, read, device, mask, independent_contexts=False):
    """Initial two contexts, then the original independent-context refinement."""
    for number, (index, row) in enumerate(rows.loc[mask].iterrows(), 1):
        image = read(row.frame_id)
        box = row[B].to_numpy(float)
        wh, center = box[2:] - box[:2], (box[:2] + box[2:]) / 2
        predictions = []
        for scale, minimum in [(4, 192), (6, 288 if independent_contexts else 192)]:
            side = max(float(wh.max() * scale), minimum)
            crop = np.clip(np.r_[center - side / 2, center + side / 2], 0, [1280, 720, 1280, 720]).astype(int)
            result = model.predict(np.asarray(image.crop(tuple(crop)))[:, :, ::-1],
                                   imgsz=800, device=device, conf=.15, iou=.5, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy() + crop[[0, 1, 0, 1]]
            conf = result.boxes.conf.cpu().numpy()
            if len(boxes):
                overlap = iou(box, boxes)[0]
                j = int(overlap.argmax())
                predictions.append((boxes[j], float(conf[j]), float(overlap[j])))
        values = [0., 0., 0.]
        if len(predictions) == 2:
            values = [min(p[2] for p in predictions), min(p[1] for p in predictions),
                      float(iou(predictions[0][0], predictions[1][0])[0, 0])]
        rows.loc[index, ['hand_iou', 'hand_conf', 'hand_stability']] = values
        if number % 100 == 0:
            print('POOL_HAND', number, int(mask.sum()), flush=True)
    return rows


def detectors(args, review=False):
    import torch
    from ultralytics import YOLO, settings
    torch.set_num_threads(2)
    settings.update({k: False for k in ['wandb', 'mlflow', 'comet', 'clearml', 'dvc', 'neptune'] if k in settings})
    paths, read = image_reader(args.raw)
    for day in DAYS:
        folder = args.out / day
        rows = pd.read_csv(folder / ('pseudo_label_ledger.csv' if review else 'candidates.csv'))
        if not review:
            targets = sorted(rows.frame_id.unique())
            model = YOLO(str(checkpoint(args, 'generic', day)))
            refs = []
            for start in range(0, len(targets), 16):
                fids = targets[start:start + 16]
                results = model.predict([str(paths[f]) for f in fids], imgsz=1280, batch=8,
                                        device=args.device, conf=.1, iou=.6, max_det=100, verbose=False)
                for fid, result in zip(fids, results):
                    scale = np.array([1280 / result.orig_shape[1], 720 / result.orig_shape[0]] * 2)
                    for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                        refs.append(dict(frame_id=fid, class_name=model.names[int(cls)], confidence=float(conf), **dict(zip(B, box * scale))))
                if start % 160 == 0:
                    print('POOL_DETECT', day, start, len(targets), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            refs = pd.DataFrame(refs, columns=['frame_id', 'class_name', 'confidence'] + B)
            refs.to_csv(folder / 'multiclass_references.csv', index=False)
            evidence = []
            for row in rows.itertuples():
                box = np.array([getattr(row, c) for c in B])
                candidates = refs[refs.frame_id.eq(row.frame_id) & refs.class_name.eq(row.class_name) & refs.confidence.ge(.6)]
                overlap = iou(box, candidates[B])[0] if len(candidates) else np.array([])
                j = int(overlap.argmax()) if len(overlap) else 0
                evidence.append(dict(teacher_iou=float(overlap[j]) if len(overlap) else 0.,
                                     teacher_conf=float(candidates.iloc[j].confidence) if len(overlap) else 0.,
                                     teacher_boxes=len(candidates), min_side=float(min(box[2:] - box[:2])),
                                     hand_iou=0., hand_conf=0., hand_stability=0.))
            rows = pd.concat([rows.reset_index(drop=True), pd.DataFrame(evidence, columns=['teacher_iou', 'teacher_conf', 'teacher_boxes', 'min_side', 'hand_iou', 'hand_conf', 'hand_stability'])], axis=1)
        mask = rows.class_name.eq('nude_hands') & rows.confidence.ge(.75)
        if review:
            mask &= rows.hand_iou.ge(.6) & rows.semantic_conf.ge(.8)
        model = YOLO(str(checkpoint(args, 'hand', day)))
        rows = hand_checks(rows, model, read, args.device, mask, independent_contexts=review)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if review:
            rows = review_candidates(rows)
        rows.to_csv(folder / ('review_ledger.csv' if review else 'detector_checks.csv'), index=False)


def automatic_selection(rows):
    rows = rows.copy()
    rows['accepted'] = (rows.confidence.ge(.85) & rows.teacher_conf.ge(.7) & rows.teacher_iou.ge(.8)
                        & rows.semantic_conf.ge(.85) & rows.min_side.ge(16))
    hands = rows.class_name.eq('nude_hands')
    rows.loc[hands, 'accepted'] = (rows.loc[hands, 'confidence'].ge(.8) & rows.loc[hands, 'min_side'].ge(16)
        & rows.loc[hands, 'semantic_conf'].ge(.85) & rows.loc[hands, 'hand_iou'].ge(.8)
        & rows.loc[hands, 'hand_conf'].ge(.65) & rows.loc[hands, 'hand_stability'].ge(.85))
    rows['overlapping_source'] = False
    for _, group in rows.groupby(['frame_id', 'class_name']):
        overlap = iou(group[B], group[B])
        np.fill_diagonal(overlap, 0)
        rows.loc[group.index, 'overlapping_source'] = overlap.max(1) > .4
    rows.loc[rows.overlapping_source, 'accepted'] = False
    return rows


def review_candidates(rows):
    rows = rows.copy()
    rows['review_candidate'] = (rows.class_name.eq('nude_hands') & rows.confidence.ge(.75)
        & rows.hand_iou.ge(.65) & rows.hand_conf.ge(.4) & rows.hand_stability.ge(.7)
        & rows.semantic_conf.ge(.8) & ~rows.overlapping_source)
    rows.loc[rows.class_name.eq('cup') & rows.accepted, 'review_candidate'] = True
    return rows


def semantic(args):
    import joblib
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    torch.set_num_threads(2)
    _, read = image_reader(args.raw)
    path = str(args.weights / 'dinov3')
    processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True, dtype=torch.float16).to(args.device).eval()
    skip = 1 + model.config.num_register_tokens
    for day in DAYS:
        folder = args.out / day
        rows = pd.read_csv(folder / 'detector_checks.csv')
        rows['semantic_conf'] = 0.
        selected = rows[rows.confidence.ge(.75) & rows.min_side.ge(16)]
        head = joblib.load(checkpoint(args, 'semantic', day))
        lookup = {c: j for j, c in enumerate(head.classes_)}
        for start in range(0, len(selected), 24):
            batch, images = selected.iloc[start:start + 24], []
            for row in batch.itertuples():
                im = read(row.frame_id)
                box = np.array([getattr(row, c) for c in B])
                wh = box[2:] - box[:2]
                for context in [0, .35]:
                    crop = np.clip(box + np.array([-1, -1, 1, 1]) * np.tile(wh, 2) * context,
                                   0, [1280, 720, 1280, 720])
                    images.append(im.crop(tuple(crop)).resize((224, 224), Image.Resampling.BICUBIC))
            inputs = processor(images=images, do_resize=False, do_center_crop=False, return_tensors='pt').to(args.device, torch.float16)
            with torch.inference_mode():
                z = model(**inputs).last_hidden_state.float()
                cls = torch.nn.functional.normalize(z[:, 0], dim=-1)
                mean = torch.nn.functional.normalize(z[:, skip:].mean(1), dim=-1)
                features = (torch.cat([cls, mean], dim=-1).cpu().numpy().reshape(len(batch), -1) / np.sqrt(2)).astype(np.float32)
            probabilities = head.predict_proba(features)
            rows.loc[batch.index, 'semantic_conf'] = [probabilities[j, lookup[c]] for j, c in enumerate(batch.class_name)]
            if start % 240 == 0:
                print('POOL_DINO', day, start, len(selected), flush=True)
        rows = automatic_selection(rows)
        rows.to_csv(folder / 'pseudo_label_ledger.csv', index=False)
        rows[rows.accepted].to_csv(folder / 'accepted_pseudo_boxes.csv', index=False)


def assemble(rows, decisions):
    """Attach annotation decisions by ID, never cached scores or coordinates."""
    if decisions.duplicated(['day', 'id']).any():
        raise ValueError('Duplicate review decisions')
    reviewed = rows[rows.review_candidate].merge(decisions[DECISIONS], on=['id', 'day'], how='left', validate='one_to_one')
    pending = reviewed[reviewed.visual_decision.isna()].copy()
    reviewed = reviewed[reviewed.visual_decision.eq('accept')].sort_values('review_index').copy()
    reviewed['geometry_safe'], reviewed['label_use'] = True, 'geometry_and_presence'
    person = rows[rows.class_name.eq('person') & rows.accepted].copy()
    person['review_index'] = np.nan
    person['visual_decision'], person['visual_reason'] = 'semantic_only', 'full_extent_not_verified'
    person['review_source'] = 'model_consensus'
    person['geometry_safe'], person['label_use'] = False, 'presence_only_not_boundary_ground_truth'
    return pd.concat([reviewed, person], ignore_index=True), pending


def finalize(args):
    decisions = pd.read_csv(args.decisions)
    summary = {}
    for day in DAYS:
        rows = pd.read_csv(args.out / day / 'review_ledger.csv')
        approved, pending = assemble(rows, decisions[decisions.day.eq(day)])
        approved.to_csv(args.out / f'approved_pseudo_boxes_{day}.csv', index=False)
        pending.to_csv(args.out / day / 'pending_review.csv', index=False)
        summary[day] = dict(candidates=len(rows), automatic=int(rows.accepted.sum()),
                            approved=len(approved), classes=approved.class_name.value_counts().to_dict(),
                            pending_review=len(pending))
    (args.out / 'manifest.json').write_text(json.dumps(dict(
        source='competition machine_predictions.csv + fresh YOLO/DINOv3 inference',
        time_exclusion_seconds=600, review_decisions=str(args.decisions),
        model_sources={day: {kind: str(checkpoint(args, kind, day)) for kind in ['generic', 'hand', 'semantic']} for day in DAYS},
        days=summary), indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def fit_semantic_heads(data, out):
    """Same training recipe as geometry_guards/fit_dino_heads.py, without evaluation."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    records = pd.read_csv(data / 'dino_records.csv')
    features = np.load(data / 'dino_features.npy').astype(np.float32).reshape(len(records), -1)
    out.mkdir(parents=True, exist_ok=True)
    for day in DAYS:
        use = records.day.eq(day) & records.kind.isin(['gt', 'short_person', 'background'])
        head = LogisticRegression(C=10, max_iter=2000, class_weight='balanced')
        head.fit(features[use], records.loc[use, 'semantic_class'], sample_weight=records.loc[use, 'weight'])
        joblib.dump(head, out / f'pool_semantic_{day}.joblib')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'cache/competition')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/machine_pool')
    parser.add_argument('--weights', type=Path, default=ROOT / 'cache/weights')
    parser.add_argument('--models', type=Path, default=ROOT / 'upstream/weights')
    parser.add_argument('--work', type=Path, help='Use the checkpoints generated by from_images.py in this upstream workspace')
    parser.add_argument('--decisions', type=Path, default=ROOT / 'upstream/annotations/pool_review_decisions.csv')
    parser.add_argument('--python-cv', type=Path, default=Path(sys.executable))
    parser.add_argument('--python-vlm', type=Path, default=ROOT / '.venv-vlm/bin/python')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stage', choices=['all', 'metadata', 'detectors', 'semantic', 'review', 'assemble', 'train-heads'], default='all')
    parser.add_argument('--training-data', type=Path, default=ROOT / 'upstream/data/geometry_guards')
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.out == ROOT or args.out.is_relative_to(ROOT / 'upstream'):
        parser.error('Use a separate output directory; bundled models and annotations are read-only inputs')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'all':
        pool_metadata(args.raw, args.out)
        for stage, python in [('detectors', args.python_cv), ('semantic', args.python_vlm), ('review', args.python_cv), ('assemble', args.python_cv)]:
            command = [str(python), str(Path(__file__).resolve()), '--stage', stage,
                       '--raw', str(args.raw), '--out', str(args.out), '--weights', str(args.weights),
                       '--models', str(args.models), '--decisions', str(args.decisions), '--device', args.device]
            if args.work is not None:
                command += ['--work', str(args.work)]
            subprocess.run(command, check=True)
    elif args.stage == 'metadata':
        pool_metadata(args.raw, args.out)
    elif args.stage in ['detectors', 'review']:
        detectors(args, review=args.stage == 'review')
    elif args.stage == 'semantic':
        semantic(args)
    elif args.stage == 'assemble':
        finalize(args)
    else:
        fit_semantic_heads(args.training_data, args.out)


if __name__ == '__main__':
    main()
