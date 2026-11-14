"""Nested cross-day calibration of multiple dense visual backbones.

No position features, no historical manual test statuses, no LB targets.
Inner models generate fit-day features, outer models predict the other day.
"""
from pathlib import Path
import sys,json,time,argparse
import numpy as np,pandas as pd,torch,cv2,joblib
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v46'))
from common46 import inputs,baseline,iou,metrics,B,CL,DAYS,CB,records
from train import Dense,decode
for p in ['inputs','features','models','logs','results/oof','results/submissions','audit']:(A/p).mkdir(parents=True,exist_ok=True)
KEYS=['small_control','vit_b','vit_l','siglip2']
def evidence(prob,r):
    b=r[B].to_numpy(float);cb=r[CB].to_numpy(float);size=cb[2:]-cb[:2];yy,xx=np.mgrid[:32,:32];x=cb[0]+(xx+.5)*size[0]/32;y=cb[1]+(yy+.5)*size[1]/32;inside=(x>=b[0])&(x<=b[2])&(y>=b[1])&(y<=b[3]);c=CL.index(r.class_name);d=decode(prob,r);means=np.array([p[inside].mean() for p in prob]);other=means[np.arange(4)!=c].max()
    return dict(iou=0. if not np.isfinite(d['overlap']) else d['overlap'],missing=float(not np.isfinite(d['overlap'])),mean=float(means[c]),coverage=float((prob[c][inside]>=.5).mean()),other_mean=float(other))
def extract_nested():
    rec=records();q,te,h=inputs();look=rec.reset_index().set_index('id')['index'].to_dict();target=np.load(ROOT/'rascar_v31/features/dense_targets.npy',mmap_mode='r');oldmanifest=json.loads((ROOT/'rascar_v45/models/nested_manifest.json').read_text());output=[];manifest=[]
    for key in KEYS:
        path=ROOT/'rascar_v31/features/dense_grids.npy' if key=='small_control' else ROOT/f'rascar_v46/features/{key}_grids.npy';features=np.load(path,mmap_mode='r')
        for fit in DAYS:
            for m in [m for m in oldmanifest if m['outer_fit_day']==fit]:
                held=set(m['held_frames']);train=rec.index[rec.split.eq('manual')&rec.frame_id.isin(m['fit_frames'])].to_numpy();assert (rec.loc[train].day==fit).all();assert not set(rec.loc[train].frame_id)&held
                model=Dense(features.shape[1]);name=f'{key}_fit{fit[-2:]}_block{m["inner_block"]}';start=time.time()
                if key=='small_control':model.load_state_dict(torch.load(ROOT/f'rascar_v45/models/inner_dense_fit{fit[-2:]}_block{m["inner_block"]}.pt',weights_only=True))
                else:
                    torch.manual_seed(3101);model=Dense(features.shape[1]);rng=np.random.default_rng(3101);count=rec.loc[train].groupby('frame_id').size();w=np.array([1/count[f] for f in rec.loc[train].frame_id]);w/=w.sum();opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01)
                    for step in range(450):
                        ix=rng.choice(train,8,replace=True,p=w);x=torch.from_numpy(np.array(features[ix],np.float32));y=torch.from_numpy(np.array(target[ix],np.float32));z=model(x);loss=F.binary_cross_entropy_with_logits(z,y,weight=1+3*y);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                    torch.save(model.state_dict(),A/f'models/{name}.pt')
                model.eval()
                for _,r in q[q.frame_id.isin(held)].iterrows():
                    i=look[r.id];rr=rec.loc[i]
                    with torch.inference_mode():prob=model(torch.from_numpy(np.array(features[i:i+1],np.float32)))[0].sigmoid().numpy()
                    output.append(dict(id=r.id,outer_fit_day=fit,role='meta_train',key=key,**evidence(prob,rr)))
                manifest.append(dict(key=key,outer_fit_day=fit,inner_block=m['inner_block'],fit_frames=m['fit_frames'],held_frames=m['held_frames'],seconds=time.time()-start));print('INNER',name,round(time.time()-start,1),flush=True)
            data=np.load(ROOT/f'rascar_v46/features/{key}_s3101_fit{fit[-2:]}_maps.npz');ml={x:i for i,x in enumerate(data['ids'])};full_maps=data['maps']
            for role,rows in [('oof',q[q.day.ne(fit)]),('test',te)]:
                for _,r in rows.iterrows():output.append(dict(id=r.id,outer_fit_day=fit,role=role,key=key,**evidence(full_maps[ml[r.id]].astype(np.float32),rec.loc[look[r.id]])))
    pd.DataFrame(output).to_csv(A/'features/nested_dense.csv',index=False);(A/'models/inner_manifest.json').write_text(json.dumps(manifest,indent=2))
def calibrate():
    q,te,h=inputs();bo,bt=baseline(q,te);e=pd.read_csv(A/'features/nested_dense.csv');data=e.pivot(index=['id','outer_fit_day','role'],columns='key',values=['iou','missing','mean','coverage','other_mean']);data.columns=[f'{k}_{v}' for v,k in data.columns];data=data.reset_index();query=pd.concat([q,te],ignore_index=True).set_index('id');columns=[c for c in data if c not in ['id','outer_fit_day','role']]
    for c in CL:data['class_'+c]=data.id.map(query.class_name).eq(c).astype(float);columns.append('class_'+c)
    data['confidence']=data.id.map(query.confidence);columns.append('confidence');data['error']=data.id.map(query.error);data['day']=data.id.map(query.day);data['frame_id']=data.id.map(query.frame_id);data['class_name']=data.id.map(query.class_name)
    assert data[columns].notna().all().all();assert data.loc[data.role.eq('test'),'error'].isna().all();data.to_csv(A/'features/meta_features.csv',index=False);predictions=[]
    for fit in DAYS:
        tr=data[data.outer_fit_day.eq(fit)&data.role.eq('meta_train')];ev=data[data.outer_fit_day.eq(fit)&data.role.ne('meta_train')];assert (tr.day==fit).all();assert (ev[ev.role.eq('oof')].day!=fit).all()
        for kind in ['logreg','catboost']:
            if kind=='logreg':
                scale=StandardScaler().fit(tr[columns]);model=LogisticRegression(C=.03,max_iter=2000).fit(scale.transform(tr[columns]),tr.error);p=model.predict_proba(scale.transform(ev[columns]))[:,1];joblib.dump((columns,scale,model),A/f'models/{kind}_fit{fit[-2:]}.joblib')
            else:
                model=CatBoostClassifier(iterations=150,depth=2,learning_rate=.03,l2_leaf_reg=10,loss_function='Logloss',random_seed=4801,thread_count=3,verbose=False,allow_writing_files=False);model.fit(tr[columns],tr.error);p=model.predict_proba(ev[columns])[:,1];model.save_model(str(A/f'models/{kind}_fit{fit[-2:]}.cbm'));(A/f'models/{kind}_columns.json').write_text(json.dumps(columns))
            predictions.append(ev[['id','outer_fit_day','role','error','day','frame_id','class_name']].assign(kind=kind,p_error=p))
    pred=pd.concat(predictions);pred.to_csv(A/'features/meta_predictions.csv',index=False);results=[]
    for kind in pred.kind.unique():
        p=pred[pred.kind.eq(kind)];oo=p[p.role.eq('oof')].set_index('id').loc[q.id].p_error.to_numpy();tt=np.stack([g.set_index('id').loc[te.id].p_error.to_numpy() for _,g in p[p.role.eq('test')].groupby('outer_fit_day')]);up=(tt>=bt).all(0);down=(tt<=bt).all(0);dt=np.where(up,tt.min(0),tt.max(0))-bt;dt[~(up|down)]=0
        for scope in ['all','person','nude_hands']:
            om=np.ones(len(q),bool) if scope=='all' else q.class_name.eq(scope).to_numpy();tm=np.ones(len(te),bool) if scope=='all' else te.class_name.eq(scope).to_numpy();s=bo.copy();t=bt.copy();s[om]+=.1*(oo[om]-bo[om]);t[tm]+=.1*dt[tm]
            for preserve in [False,True]:
                so=s.copy();st=t.copy()
                if preserve:
                    for cls in CL if scope=='all' else [scope]:
                        for rows,base,dest in [(q,bo,so),(te,bt,st)]:
                            ix=np.flatnonzero(rows.class_name.eq(cls));order=np.argsort(dest[ix],kind='stable');dest[ix[order]]=np.sort(base[ix])
                name=f'{kind}_{scope}_w10'+('_scale_preserved' if preserve else '')
                q.assign(score=so).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=st).to_csv(A/f'results/submissions/{name}.csv',index=False);results.append(dict(name=name,**metrics(q,so),delta=AP(q.error,so)-AP(q.error,bo),standalone_ap=AP(q.error,oo),changed_test=int((abs(st-bt)>1e-9).sum())))
    df=pd.DataFrame(results).sort_values('ap',ascending=False);df.to_csv(A/'results/experiments.csv',index=False);print(df.to_string(index=False))
def main():
    p=argparse.ArgumentParser();p.add_argument('--calibrate-only',action='store_true');args=p.parse_args();torch.set_num_threads(3);cv2.setNumThreads(1)
    if not args.calibrate_only:extract_nested()
    calibrate()
if __name__=='__main__':main()
