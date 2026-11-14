"""Query-local hand/non-hand classifier with real, train-only hard negatives.

The fresh pool review is evaluation-only. No test labels, no score fitting on
the reverse OOF fold, no claim that hand presence proves valid geometry.
"""
from experiments import *
import os,time,joblib,cv2,torch
from transformers import AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score

CB=['crop_x1','crop_y1','crop_x2','crop_y2']
torch.set_num_threads(3);cv2.setNumThreads(1)

def local_feature(grid,r):
    cb=r[CB].to_numpy(float);b=(r[B].to_numpy(float)-np.r_[cb[:2],cb[:2]])/np.r_[cb[2:]-cb[:2],cb[2:]-cb[:2]]
    yy,xx=np.mgrid[:32,:32];xx=(xx+.5)/32;yy=(yy+.5)/32
    mask=(xx>=b[0])&(xx<=b[2])&(yy>=b[1])&(yy<=b[3])
    assert mask.any()
    g=np.asarray(grid,np.float32);inside=g[:,mask].mean(1);outside=g[:,~mask].mean(1)
    # Position is deliberately excluded: learn appearance, not a test-location rule.
    return np.r_[inside,inside-outside].astype(np.float32)

def new_features():
    fresh=pd.read_csv(A/'inputs/fresh_pool_candidates.csv')
    negatives=pd.read_csv(ROOT/'rascar_v33/audit/rare_hand_visual_decisions.csv')
    negatives=negatives[negatives.decision.eq('reject_fabric')].rename(columns={'detection_id':'id'}).copy()
    negatives['split']='old_negative';fresh['split']='fresh_pool'
    rec=pd.concat([fresh,negatives],ignore_index=True)
    for i,r in rec.iterrows():
        b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);center=(b[:2]+b[2:])/2
        rec.loc[i,CB]=np.r_[np.maximum(0,np.floor(center-1.2*wh)),np.minimum([1280,720],np.ceil(center+1.2*wh))]
    rec.to_csv(A/'inputs/semantic_extra_records.csv',index=False)
    (A/'features').mkdir(exist_ok=True)
    dest=A/'features/semantic_extra.npy'
    if dest.exists():
        x=np.load(dest);assert len(x)==len(rec);return rec,x
    model=AutoModel.from_pretrained(str(ROOT/'rascar_v31/model_cache/dinov2-small'),local_files_only=True,trust_remote_code=False).eval()
    mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1);out=[]
    for off in range(0,len(rec),4):
        rr=rec.iloc[off:off+4];batch=[]
        for _,r in rr.iterrows():
            im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));cb=r[CB].to_numpy(int)
            crop=im[cb[1]:cb[3],cb[0]:cb[2],::-1]
            batch.append(cv2.resize(crop,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
        x=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std
        with torch.inference_mode():grid=model(pixel_values=x).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,32,32).numpy()
        out.extend([local_feature(g,r) for g,(_,r) in zip(grid,rr.iterrows())])
        print('NEW_CROPS',off+len(rr),len(rec),flush=True)
    out=np.stack(out);np.save(dest,out);return rec,out

def prepare():
    frames=pd.read_csv(ROOT/'rascar_v22/inputs/manual_boxes.csv')
    gt={f:d[d.class_name.eq('nude_hands')][B].to_numpy(float) for f,d in frames.groupby('frame_id')}
    recs=[];features=[]
    banks=[('rascar_v31/inputs/dense_records.csv','rascar_v31/features/dense_grids.npy'),
           ('rascar_v33/inputs/empty_records.csv','rascar_v33/features/empty_grids.npy'),
           ('rascar_v33/inputs/pseudo_records.csv','rascar_v33/features/pseudo_grids.npy'),
           ('rascar_v33/inputs/rare_records.csv','rascar_v33/features/rare_grids.npy')]
    for rp,fp in banks:
        rr=pd.read_csv(ROOT/rp);arr=np.load(ROOT/fp,mmap_mode='r')
        for _,r in rr.iterrows():
            target=np.nan;weight=1.
            if r.split=='test':
                if r.class_name!='nude_hands':continue
            elif r.split.startswith('pool_'):
                if r.class_name!='nude_hands':continue
                target=1.;weight=.3
            else:
                g=gt[r.frame_id];b=r[B].to_numpy(float)
                ov=iou(b,g).ravel()
                inter=np.maximum(0,np.minimum(b[2:],g[:,2:])-np.maximum(b[:2],g[:,:2])).prod(1)
                coverage=inter/np.maximum((g[:,2:]-g[:,:2]).prod(1),1)
                if r.class_name=='nude_hands' and r.split=='manual':target=1.
                elif len(g) and ov.max()>=.3:target=1.
                elif not len(g) or (ov.max()<.02 and coverage.max()<.05):target=0.
                elif r.split!='oof' or r.class_name!='nude_hands':continue
            z=r.to_dict();z['semantic_target']=target;z['source_weight']=weight
            recs.append(z);features.append(local_feature(arr[int(r.feature_index)],r))
    extra,x=new_features()
    for j,r in extra.iterrows():
        z=r.to_dict();z['semantic_target']=0. if r.split=='old_negative' else np.nan;z['source_weight']=1.
        recs.append(z);features.append(x[j])
    rec=pd.DataFrame(recs);x=np.stack(features);assert rec.id.is_unique
    rec.to_csv(A/'inputs/semantic_records.csv',index=False);np.save(A/'features/semantic_vectors.npy',x)
    return rec,x

def main():
    rec,x=prepare();results=[];manifest=[];evalrows=[]
    (A/'models').mkdir(exist_ok=True)
    fresh_ids=set(rec.loc[rec.split.eq('fresh_pool'),'frame_id'])
    for cval in [.03,.1]:
        preds={}
        for fit in DAYS:
            train=rec.day.eq(fit)&rec.semantic_target.notna()&~rec.split.isin(['test','fresh_pool'])
            assert not set(rec.loc[train,'frame_id'])&fresh_ids
            weights=rec.loc[train,'source_weight']/rec.loc[train].groupby('frame_id').id.transform('size')
            weights*=train.sum()/weights.sum()
            model=make_pipeline(StandardScaler(),LogisticRegression(C=cval,class_weight='balanced',max_iter=1000,random_state=3411))
            model.fit(x[train],rec.loc[train,'semantic_target'],logisticregression__sample_weight=weights.to_numpy())
            joblib.dump(model,A/f'models/hand_presence_c{cval}_fit{fit[-2:]}.joblib')
            prob=model.predict_proba(x)[:,1];preds[fit]=pd.Series(prob,index=rec.id)
            manifest.append(dict(C=cval,fit_day=fit,n=int(train.sum()),frames=int(rec.loc[train].frame_id.nunique()),
                positives=int(rec.loc[train,'semantic_target'].sum()),fit_frames=sorted(rec.loc[train,'frame_id'].unique()),
                fresh_pool_in_fit=False,test_in_fit=False))
            use=rec.split.eq('test')|((rec.day.ne(fit))&rec.split.isin(['oof','fresh_pool']))
            e=rec.loc[use,['id','frame_id','day','class_name','split','semantic_target']].copy()
            e['C']=cval;e['fit_day']=fit;e['p_hand']=prob[use];evalrows.append(e)
        oo=np.ones(len(q));tt=np.ones(len(te));oh=q.class_name.eq('nude_hands').to_numpy();th=te.class_name.eq('nude_hands').to_numpy()
        for fit in DAYS:
            m=oh&q.day.ne(fit).to_numpy();oo[m]=preds[fit].loc[q.loc[m,'id']]
        tt[th]=np.maximum.reduce([preds[fit].loc[te.loc[th,'id']].to_numpy() for fit in DAYS])
        for cutoff in [.05,.15]:
            for weight in [.25,.5]:
                scores={}
                for split,p,mask in [('oof',oo,oh),('test',tt,th)]:
                    active=mask&(p<cutoff);target=1-p
                    scores[split]=BASE[split]+weight*np.maximum(target-BASE[split],0)*active
                results.append(save(f'hand_semantic_c{cval}_p{cutoff}_w{weight}',scores,'semantic_veto',
                    'Query-local DINO inside and contrast features; train-day manual/pseudo positives, manual negatives and 3 previously rejected fabric crops. New pool review evaluation-only. Both fit days must predict absence on test.'))
    pd.DataFrame(results).to_csv(A/'results/semantic_experiments.csv',index=False)
    ev=pd.concat(evalrows,ignore_index=True);ev.to_csv(A/'audit/semantic_predictions.csv',index=False)
    (A/'models/semantic_manifest.json').write_text(json.dumps(manifest,indent=2))
    labels=json.loads((A/'inputs/fresh_pool_presence_review.json').read_text());fresh=pd.read_csv(A/'inputs/fresh_pool_candidates.csv')
    fresh['presence']=np.nan
    fresh.loc[fresh.review_index.isin(labels['present_review_indices']),'presence']=1
    fresh.loc[fresh.review_index.isin(labels['absent_review_indices']),'presence']=0
    evaluated=ev[ev.split.eq('fresh_pool')].merge(fresh[['id','review_index','presence','stratum']],on='id')
    evaluated.to_csv(A/'audit/fresh_pool_semantic_predictions.csv',index=False)
    summaries=[]
    for (c,day),d in evaluated.groupby(['C','day']):
        d=d[d.presence.notna()];y=1-d.presence.to_numpy();p=1-d.p_hand.to_numpy()
        summaries.append(dict(C=c,day=day,n=len(d),nonhands=int(y.sum()),presence_detection_ap=float(average_precision_score(y,p)),
            true_hands_flagged_p005=int(((d.presence==1)&(d.p_hand<.05)).sum()),
            true_hands_flagged_p015=int(((d.presence==1)&(d.p_hand<.15)).sum()),
            nonhands_flagged_p005=int(((d.presence==0)&(d.p_hand<.05)).sum()),
            nonhands_flagged_p015=int(((d.presence==0)&(d.p_hand<.15)).sum())))
    pd.DataFrame(summaries).to_csv(A/'results/fresh_pool_semantic.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__':main()
