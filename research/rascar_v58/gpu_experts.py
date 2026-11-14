"""V58: new day-held-out experts. No test annotations or manual flags used."""
from pathlib import Path
import numpy as np,pandas as pd,torch,time,json,sys
from tabicl import TabICLClassifier
A=Path('/kaggle/working/rascar_v56');O=Path('/kaggle/working/rascar_v58')
for d in ['features','models','audit','logs']:(O/d).mkdir(exist_ok=True,parents=True)
rows=pd.read_csv(A/'inputs/rows.csv'); torch.set_num_threads(2)
full=pd.read_csv(A/'features/sam2_full.csv').set_index('id').loc[rows.id].drop(columns=[],errors='ignore').reset_index(drop=True)
crop=pd.read_csv(A/'features/sam2_crop.csv').set_index('id').loc[rows.id].reset_index(drop=True)
q=pd.read_csv(A/'inputs/q.csv');t=pd.read_csv(A/'inputs/t.csv');meta=pd.concat([q,t],ignore_index=True).set_index('id').loc[rows.id].reset_index(drop=True)
M=meta[['confidence']].copy()
for key,scale in [('x1',1280),('y1',720),('x2',1280),('y2',720)]:M['meta_'+key]=meta[key]/scale
M['meta_width']=(meta.x2-meta.x1)/1280;M['meta_height']=(meta.y2-meta.y1)/720
M['meta_aspect']=np.log((meta.x2-meta.x1)/(meta.y2-meta.y1))
M['meta_count']=meta.groupby(['frame_id','class_name']).confidence.transform('size')
checkpoint=str(A/'models/tabicl-classifier-v1.1-20250506.ckpt');manifest=[];start=time.time()
family=sys.argv[1] if len(sys.argv)>1 else 'both_anchor'
for seed in [5801,5802,5803]:
 name=f'{family}_s{seed}';dest=O/f'features/{name}_predictions.csv'
 if dest.exists():continue
 pred=[]
 for fit in sorted(rows[rows.split.eq('oof')].day.unique()):
  X=pd.read_csv(A/f'features/anchor_fit{fit[-2:]}_bank.csv')
  if 'both' in family:X=pd.concat([X,full,crop],axis=1)
  if 'meta' in family:X=pd.concat([X,M],axis=1)
  X=X.fillna(0)
  classes=sorted(rows.class_name.unique()) if family.startswith('class_') else ['all']
  for cls in classes:
   tr=rows.split.eq('oof')&rows.day.eq(fit);ev=~tr
   if cls!='all':tr &= rows.class_name.eq(cls);ev &= rows.class_name.eq(cls)
   assert set(rows.loc[tr,'frame_id']).isdisjoint(rows.loc[ev,'frame_id'])
   if rows.loc[tr,'error'].nunique()<2:raise ValueError(('degenerate',fit,cls))
   model=TabICLClassifier(n_estimators=16,model_path=checkpoint,allow_auto_download=False,device=('cuda:1' if family.startswith('class') else 'cuda:0'),n_jobs=2,batch_size=2,random_state=seed,use_amp=False)
   cols=[c for c in X if cls=='all' or c!='class_name']
   model.fit(X.loc[tr,cols],rows.loc[tr,'error'].astype(int));p=model.predict_proba(X.loc[ev,cols])[:,1]
   pred.append(rows.loc[ev,['id','split']].assign(fit_day=fit,p_error=p))
   model.save(O/f'models/{name}_fit{fit[-2:]}_{cls}.pkl',save_model_weights=False,save_training_data=True,save_kv_cache=False)
   manifest.append(dict(name=name,fit_day=fit,cls=cls,train_ids=rows.loc[tr,'id'].tolist(),columns=cols,seconds=time.time()-start))
   print('DONE',name,fit,cls,round(time.time()-start,1),flush=True)
 d=pd.concat(pred);assert not d.duplicated(['id','split','fit_day']).any();d.to_csv(dest,index=False)
 (O/f'audit/{family}_manifest.json').write_text(json.dumps(manifest,indent=2))
print('EXPERTS_COMPLETE',family,flush=True)
