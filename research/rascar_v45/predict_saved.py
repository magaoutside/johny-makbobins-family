"""Reproduce the V45 candidate from saved test features and fitted rank heads.

Requires project V31 baseline and V45 cache/models. Does not retrain, read
manual test annotations, or contact Kaggle. For feature recomputation see report.
"""
from common45 import *
import argparse,joblib,torch
from torch import nn
p=argparse.ArgumentParser();p.add_argument('--output',default=str(A/'submission_v45_person_nested_rank_RESEARCH.csv'));args=p.parse_args()
te=pd.read_csv(D/'test.csv');base=pd.read_csv(ROOT/'rascar_v31/submission_v31_dense_oof098784.csv').set_index('id').loc[te.id,'score'].to_numpy()
f=pd.read_csv(A/'features/nested_quality.csv');f=f[f.role.eq('test')];changes=[]
for fit in DAYS:
    cols,scale=joblib.load(A/f'models/cross_class_rank_scale_fit{fit[-2:]}.joblib');m=nn.Linear(len(cols),1);m.load_state_dict(torch.load(A/f'models/cross_class_rank_fit{fit[-2:]}.pt',map_location='cpu',weights_only=True));m.eval();d=f[f.outer_fit_day.eq(fit)].set_index('id').loc[te.id]
    with torch.inference_mode():pred=m(torch.tensor(scale.transform(d[cols]),dtype=torch.float32))[:,0].sigmoid().numpy()
    changes.append(pred-np.clip((.85-d.overlap.to_numpy())/.5,0,1))
changes=np.stack(changes);ok=np.isfinite(changes).all(0)&((changes>=0).all(0)|(changes<=0).all(0))&te.class_name.eq('person').to_numpy();delta=np.where((changes>=0).all(0),np.min(changes,0),np.max(changes,0));new=base.copy();new[ok]=np.clip(base[ok]+.1*delta[ok],0,1)
ix=np.flatnonzero(te.class_name.eq('person'));order=ix[np.argsort(new[ix],kind='stable')];new[order]=np.sort(base[ix]);out=te[['id']].assign(score=new)
assert len(out)==362 and out.id.is_unique and out.score.between(0,1).all();assert np.array_equal(new[~te.class_name.eq('person')],base[~te.class_name.eq('person')])
out.to_csv(args.output,index=False);expected=pd.read_csv(A/'results/submissions/nested_cross_class_rank_person_w10_scale_preserved.csv').set_index('id').loc[te.id,'score'].to_numpy();assert np.array_equal(new,expected)
print(json.dumps(dict(output=args.output,rows=len(out),sha256=hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),exact_saved_candidate_match=True),indent=2))
