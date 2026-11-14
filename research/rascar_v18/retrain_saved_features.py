"""Verify both released learned additions by refitting saved opposite-day features."""
from pathlib import Path
import json
import numpy as np,pandas as pd,joblib
from catboost import CatBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
A=Path(__file__).resolve().parent;checks=[];days=['2026-10-25','2026-10-26']
for fit,held in zip(days,days[::-1]):
    rows=pd.read_csv(A/f'hand_geometry_training_{fit}.csv');x=pd.read_csv(A/f'hand_geometry_train_features_{fit}.csv')
    v=pd.read_csv(A/f'hand_geometry_held_features_{fit}.csv');t=pd.read_csv(A/f'hand_geometry_test_features_{fit}.csv');cols=[c for c in x if c!='id']
    assert rows.day.eq(fit).all() and rows.id.tolist()==x.id.tolist()
    queries=pd.read_csv(A/'replay_inputs/oof.csv').set_index('id')
    assert queries.loc[v.id,'day'].eq(held).all()
    assert not set(rows.frame_id)&set(queries.loc[v.id,'frame_id'])
    w=rows.weight.to_numpy().copy();w[rows.kind.eq('query_train')]*=2
    model=CatBoostClassifier(iterations=300,depth=3,learning_rate=.03,l2_leaf_reg=15,random_seed=181,verbose=False,thread_count=2,allow_writing_files=False)
    model.fit(x[cols],rows.error,sample_weight=w);saved=joblib.load(A/f'hand_geometry_catboost_{fit}.joblib')
    for split,features in [('held',v),('test',t)]:
        diff=float(abs(model.predict_proba(features[cols])-saved.predict_proba(features[cols])).max());assert diff<1e-10
        checks.append(dict(model='hand_catboost',fit=fit,split=split,max_diff=diff))
    x=pd.read_csv(A/'person_body_training_features.csv');y=pd.read_csv(A/'person_body_training_targets.csv');m=y.day.eq(fit)
    model=make_pipeline(StandardScaler(),Ridge(alpha=100));model.fit(x[m],y.loc[m,['x1','y1','x2','y2']])
    saved=joblib.load(A/f'person_body_ridge_{fit}.joblib')
    for split in ['oof','test']:
        f=pd.read_csv(A/f'person_body_query_features_{split}.csv');diff=float(abs(model.predict(f)-saved.predict(f)).max());assert diff<1e-10
        checks.append(dict(model='body_ridge',fit=fit,split=split,max_diff=diff))
(A/'model_refit_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
