"""Delete-one-training-frame ensembles, opposite-day validation, no test labels."""
from pathlib import Path
import pandas as pd,numpy as np,torch,sys,time,json
from tabicl import TabICLClassifier
A=Path('/kaggle/working/rascar_v56');O=Path('/kaggle/working/rascar_v58');family=sys.argv[1];rows=pd.read_csv(A/'inputs/rows.csv');torch.set_num_threads(2)
new=[]
for mode in (['full'] if family=='full_anchor' else ['crop'] if family=='crop_anchor' else []):new.append(pd.read_csv(A/f'features/sam2_{mode}.csv').set_index('id').loc[rows.id].reset_index(drop=True))
start=time.time();pred=[];manifest=[]
for fit in sorted(rows[rows.split.eq('oof')].day.unique()):
 X=pd.concat([pd.read_csv(A/f'features/anchor_fit{fit[-2:]}_bank.csv')]+new,axis=1).fillna(0)
 alltr=rows.split.eq('oof')&rows.day.eq(fit);ev=~alltr
 frames=sorted(rows.loc[alltr,'frame_id'].unique())
 for j,frame in enumerate(frames):
  tr=alltr&rows.frame_id.ne(frame)
  model=TabICLClassifier(n_estimators=16,model_path=str(A/'models/tabicl-classifier-v1.1-20250506.ckpt'),allow_auto_download=False,device=('cuda:1' if family=='anchor' else 'cuda:0'),n_jobs=2,batch_size=2,random_state=5801,use_amp=False)
  model.fit(X.loc[tr],rows.loc[tr,'error'].astype(int));p=model.predict_proba(X.loc[ev])[:,1]
  part=rows.loc[ev,['id','split','class_name']].assign(fit_day=fit,excluded_frame=frame,p_error=p)
  for c in rows.class_name.unique():
   train=rows[tr&rows.class_name.eq(c)];full=rows[alltr&rows.class_name.eq(c)]
   part.loc[part.class_name.eq(c),'bag_prior']=(train.error.sum()+1)/(len(train)+2)
   part.loc[part.class_name.eq(c),'full_prior']=(full.error.sum()+1)/(len(full)+2)
  pred.append(part)
  # Save training data per jackknife model for exact reproduction (no weight duplication).
  model.save(O/f'models/jk_{family}_fit{fit[-2:]}_f{j:02d}.pkl',save_model_weights=False,save_training_data=True,save_kv_cache=False)
  manifest.append(dict(family=family,fit_day=fit,excluded_frame=frame,train_ids=rows.loc[tr,'id'].tolist(),columns=list(X)))
  pd.concat(pred).to_csv(O/f'features/jk_{family}_predictions.csv',index=False)
  print('JACKKNIFE',family,fit,j+1,len(frames),round(time.time()-start,1),flush=True)
(O/f'audit/jk_{family}_manifest.json').write_text(json.dumps(manifest,indent=2))
print('JACKKNIFE_COMPLETE',family,flush=True)
