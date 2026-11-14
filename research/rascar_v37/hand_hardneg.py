"""Add clear V36/V37 pool hand-presence labels to their own training day.

The original blind reviews remain immutable; their role is explicitly
promoted here. Report opposite-day pool diagnostics, not a fresh holdout.
"""
from common37 import *
import cv2,torch,joblib,time
from transformers import AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
torch.set_num_threads(3);cv2.setNumThreads(1)

def vectors(rec,filename='pool_context_vectors.npy'):
    dest=A/'features'/filename
    if dest.exists():
        x=np.load(dest)
        assert len(x)==len(rec), f'Feature cache row mismatch: {dest}'
        return x
    model=AutoModel.from_pretrained(str(ROOT/'rascar_v31/model_cache/dinov2-small'),local_files_only=True).eval()
    out=[];mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
    yy,xx=np.mgrid[:32,:32];xy=np.stack([(xx+.5)/32,(yy+.5)/32],axis=-1)
    for off in range(0,len(rec),4):
        batch=[];boxes=[]
        for _,r in rec.iloc[off:off+4].iterrows():
            b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2
            lo=np.maximum(0,np.floor(cc-1.2*wh)).astype(int);hi=np.minimum([1280,720],np.ceil(cc+1.2*wh)).astype(int)
            im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720))[lo[1]:hi[1],lo[0]:hi[0],::-1]
            batch.append(cv2.resize(im,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy());boxes.append((b-np.r_[lo,lo])/np.r_[hi-lo,hi-lo])
        with torch.inference_mode():z=model(pixel_values=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,32,32).numpy()
        for grid,b in zip(z,boxes):
            m=((xy>=b[:2])&(xy<=b[2:])).all(2);inside=grid[:,m].mean(1);outside=grid[:,~m].mean(1);out.append(np.r_[inside,inside-outside])
    x=np.stack(out);np.save(dest,x);return x

def main():
    rec=pd.read_csv(ROOT/'rascar_v36/inputs/pool_augmented_semantic_records.csv');x=np.load(ROOT/'rascar_v34/features/semantic_vectors.npy')
    r36=pd.read_csv(ROOT/'rascar_v36/inputs/blind_pool_candidates.csv');r37=pd.read_csv(A/'inputs/blind_pool.csv');r37=r37[r37.class_name.eq('nude_hands')].copy()
    lab36=json.loads((ROOT/'rascar_v36/inputs/blind_pool_presence_review.json').read_text());lab37=json.loads((A/'inputs/blind_review.json').read_text())
    for r,lab,pk,nk in [(r36,lab36,'present_review_indices','absent_review_indices'),(r37,lab37,'present','absent')]:
        r['semantic_target']=np.nan;r.loc[r.review_index.isin(lab[pk]),'semantic_target']=1;r.loc[r.review_index.isin(lab[nk]),'semantic_target']=0
        r['source_weight']=1.;r['split']='promoted_pool'
    r36['review_source']='v36';r37['review_source']='v37'
    xx=np.concatenate([np.load(ROOT/'rascar_v36/features/blind_pool_dino.npy'),vectors(r37)])
    pool=pd.concat([r36,r37],ignore_index=True);pool.to_csv(A/'inputs/promoted_pool_presence.csv',index=False)
    rec=pd.concat([rec,pool],ignore_index=True);x=np.concatenate([x,xx]);assert rec.id.is_unique
    predictions={};evalrows=[];manifest=[];metrics=[]
    for fit in DAYS+['all']:
        train=(rec.day.isin(DAYS) if fit=='all' else rec.day.eq(fit))&rec.semantic_target.notna()&~rec.split.isin(['test','fresh_pool'])
        weights=rec.loc[train,'source_weight']/rec.loc[train].groupby('frame_id').id.transform('size');weights*=train.sum()/weights.sum()
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=1500,random_state=3701));model.fit(x[train],rec.loc[train,'semantic_target'],logisticregression__sample_weight=weights)
        joblib.dump(model,A/f'models/hand_context_fit{fit[-2:]}.joblib');predictions[fit]=pd.Series(model.predict_proba(x)[:,1],index=rec.id)
        manifest.append(dict(fit_day=fit,fit_frames=sorted(rec.loc[train,'frame_id'].unique()),n=int(train.sum()),new_pool_crops=int((train&rec.split.eq('promoted_pool')).sum()),test_in_fit=False))
        if fit!='all':
            use=rec.split.eq('promoted_pool')&rec.day.ne(fit)&rec.semantic_target.notna();d=rec.loc[use].copy();assert not set(d.frame_id)&set(rec.loc[train,'frame_id'])
            d['p_present']=predictions[fit].loc[d.id].to_numpy();d['fit_day']=fit;evalrows.append(d)
    pd.concat(evalrows).to_csv(A/'audit/promoted_pool_crossday_predictions.csv',index=False)
    (A/'models/hand_context_manifest.json').write_text(json.dumps(manifest,indent=2))
    summaries=[]
    for (source,day),d in pd.concat(evalrows).groupby(['review_source','day']):
        summaries.append(dict(review_source=source,day=day,n=len(d),negative=int(d.semantic_target.eq(0).sum()),false_rejects=int((d.semantic_target.eq(1)&d.p_present.lt(.15)).sum()),true_rejects=int((d.semantic_target.eq(0)&d.p_present.lt(.15)).sum())))
    pd.DataFrame(summaries).to_csv(A/'results/promoted_pool_crossday.csv',index=False);print(pd.DataFrame(summaries).to_string(index=False))
    for mode in ['daily_consensus','full_fit']:
        probs={s:np.ones(len(rows)) for s,rows in ROWS.items()}
        for fit in DAYS:
            mask=q.day.ne(fit)&q.class_name.eq('nude_hands');probs['oof'][mask]=predictions[fit].loc[q.loc[mask,'id']]
        mask=te.class_name.eq('nude_hands');ids=te.loc[mask,'id']
        probs['test'][mask]=predictions['all'].loc[ids] if mode=='full_fit' else np.maximum.reduce([predictions[fit].loc[ids].to_numpy() for fit in DAYS])
        for cutoff in [.05,.15]:
            for weight in [.25,.5]:
                scores={s:BASE[s]+weight*np.maximum(1-p-BASE[s],0)*(p<cutoff)*ROWS[s].class_name.eq('nude_hands').to_numpy() for s,p in probs.items()}
                metrics.append(save(f'hand_context_{mode}_p{cutoff}_w{weight}',scores,'New train-pool presence labels fit only their own day. OOF and pool diagnostics held by day; no test labels. Full-fit is a final-training variant, not separate OOF evidence.'))
        for s,rows in ROWS.items():rows.assign(p_present=probs[s]).to_csv(A/f'features/hand_context_{mode}_{s}.csv',index=False)
    pd.DataFrame(metrics).to_csv(A/'results/hand_context_experiments.csv',index=False)

if __name__=='__main__':main()
