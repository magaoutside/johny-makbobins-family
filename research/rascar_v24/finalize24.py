from common24 import *
from collections import Counter
import shutil,joblib
R=A/'release';R.mkdir(exist_ok=True)
models={'cup_context':(ROOT/'rascar_v23/models','combined_catboost_cup'),
        'cup_boundary':(A/'models','context_cb_cup'),
        'hand_boundary':(A/'models','context_trees_nude_hands')}
for fit,held in zip(DAYS,DAYS[::-1]):
    for expert,(folder,name) in models.items():
        path=folder/f'{name}_{fit}.joblib';bundle=joblib.load(path)
        assert bundle['fit_day']==fit and not set(bundle['training_frames'])&set(q.loc[q.day.eq(held),'frame_id'])
        shutil.copyfile(path,R/f'{expert}_{fit}.joblib')
    for split in ['oof','test']:
        source=ROOT/f'rascar_v23/release/{split}_features_{fit}.csv'
        shutil.copyfile(source,R/f'v23_{split}_{fit}.csv')
for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
    d=rows[['id','frame_id','class_name']].copy();d['day']=rows.day.to_numpy() if 'day' in rows else 'test';d['baseline']=base
    d['cup_guard']=guard(split,rows,'cup').astype(int);d['hand_guard']=guard(split,rows,'nude_hands').astype(int)
    f=align(A/f'features/instances_{split}.csv',rows)
    d['geometry_guard']=(f.assigned.eq(1)&f.rival_margin.ge(.1)&f.reference_iou.ge(.65)&f.ref_conf.ge(.55)).astype(int).to_numpy()
    d.to_csv(R/f'{split}_inputs.csv',index=False)
    shutil.copyfile(A/f'features/boundary_{split}.csv',R/f'v24_{split}.csv')
    for variant,experiment in [('candidate','cup_experts_minimum'),('hand_research','cup_min_guarded_hand_research')]:
        source=A/(f'results/oof/{experiment}.csv' if split=='oof' else f'results/submissions/submission_{experiment}.csv')
        result=pd.read_csv(source)[['id','score']];assert result.id.equals(rows.id.reset_index(drop=True))
        assert len(result)==len(rows) and result.id.nunique()==len(rows) and result.score.between(0,1).all()
        result.to_csv(R/f'{variant}_{split}_expected.csv',index=False)
        if split=='test':shutil.copyfile(source,A/f'submission_v24_{variant}.csv')

records=[json.loads(p.read_text()) for p in (A/'results').glob('*.json')]
pd.DataFrame(records).sort_values('pooled_ap',ascending=False).to_csv(A/'results/experiment_registry.csv',index=False)
candidate=json.loads((A/'results/cup_experts_minimum.json').read_text());research=json.loads((A/'results/cup_min_guarded_hand_research.json').read_text())
previous=json.loads((ROOT/'rascar_v23/results/combined_catboost_cup_rescue_w0.4.json').read_text())
current_oof=align(A/'results/oof/cup_experts_minimum.csv',q,'score');previous_oof=align(ROOT/'rascar_v23/results/oof/combined_catboost_cup_rescue_w0.4.csv',q,'score')
rng=np.random.default_rng(2401);groups=[[np.flatnonzero(q.frame_id.eq(fid)) for fid in q.loc[q.day.eq(day),'frame_id'].unique()] for day in DAYS];deltas=[]
for _ in range(2000):
    indices=np.concatenate([np.concatenate([group[k] for k in rng.integers(0,len(group),len(group))]) for group in groups])
    yy=q.error.to_numpy()[indices];deltas.append(AP(yy,current_oof[indices])-AP(yy,previous_oof[indices]))
bootstrap=dict(draws=2000,unit='frame within day',vs='V23 cup candidate',delta=candidate['pooled_ap']-previous['pooled_ap'],
               interval95=list(map(float,np.quantile(deltas,[.025,.975]))),caveat='Descriptive, not independent; experiment selection is not accounted for.')
summary=dict(baseline=metric(q,base_oof),confirmed_best_lb=.95610,previous_v23=previous,candidate=candidate,hand_research=research,
    experiments=len(records),families=dict(Counter(r['details']['type'] for r in records)),
    trained_boundary_heads=24,trained_offset_regressors=4,training_rows=len(pd.read_csv(A/'inputs/boundary_training.csv')),
    bootstrap=bootstrap,submitted_to_leaderboard=False,test_labels_used_for_training=False,test_audit_used_for_selection=True,
    primary_excludes_inherited_single_oof_correction=True,raw_pool_used_for_new_pixel_training=False,
    limitations=['35 OOF frames, repeated selection','Test visual labels are subjective','Short occluded person labels conflict with some manual annotation conventions',
                 'Guarded hand branch has no OOF coverage','New CV models do not retrain the upstream frozen detector'])
(A/'summary.json').write_text(json.dumps(summary,indent=2))
for name in ['cup_experts_minimum','cup_min_guarded_hand_research']:
    d=pd.read_csv(A/f'results/reviews/{name}.csv');d[d.delta.abs()>1e-9].to_csv(A/f'audit/{name}_changes.csv',index=False)
manifest={str(p.relative_to(A)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*') if p.is_file()}
manifest['submission_v24_candidate.csv']=hashlib.sha256((A/'submission_v24_candidate.csv').read_bytes()).hexdigest()
manifest['submission_v24_hand_research.csv']=hashlib.sha256((A/'submission_v24_hand_research.csv').read_bytes()).hexdigest()
(A/'release_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(experiments=len(records),bootstrap=bootstrap,candidate=candidate['pooled_ap'],research_remaining=research['remaining_known']),indent=2))
