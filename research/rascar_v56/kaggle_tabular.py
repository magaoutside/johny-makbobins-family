from common56 import *
from tabicl import TabICLClassifier
from tabicl._sklearn.base import TabICLBaseEstimator
import torch,pickle
torch.set_num_threads(2);q,t,h=inputs();bo,bt=incumbent(q,t);rows=pd.read_csv(A/'inputs/rows.csv');results=[];checks=[]
checkpoint=str(A/'models/tabicl-classifier-v1.1-20250506.ckpt')
original_setstate=TabICLBaseEstimator.__setstate__
def relocated(self,state):
    state['model_path']=checkpoint;state['device']='cuda:0';state['n_jobs']=2
    original_setstate(self,state)
TabICLBaseEstimator.__setstate__=relocated
for seed,n in [(5601,8),(5602,16),(5603,16)]:
    name=f'tabiclv1_anchor_n{n}_s{seed}';cached=pd.read_csv(A/f'features/{name}_predictions.csv')
    for fit in DAYS:
        tr=rows.split.eq('oof')&rows.day.eq(fit);ev=~tr;X=pd.read_csv(A/f'features/anchor_fit{fit[-2:]}_bank.csv')
        model=TabICLClassifier.load(A/f'models/{name}_fit{fit[-2:]}.pkl',device='cuda:0');p=model.predict_proba(X.loc[ev])[:,1]
        expected=cached[cached.fit_day.eq(fit)].set_index('id').loc[rows.loc[ev,'id'],'p_error'].to_numpy();err=float(abs(p-expected).max())
        checks.append(dict(name=name,fit_day=fit,max_error=err,train_validation_frames_disjoint=set(rows.loc[tr,'frame_id']).isdisjoint(rows.loc[ev,'frame_id'])))
        assert err<.001 and checks[-1]['train_validation_frames_disjoint'],checks[-1]
        print('REPLAY',checks[-1],flush=True)
(A/'audit/kaggle_candidate_replay.json').write_text(json.dumps(checks,indent=2))
# Further seed confirmation for independent SAM2 person/cup experts.
full=pd.read_csv(A/'features/sam2_full.csv').set_index('id').loc[rows.id].reset_index(drop=True)
crop=pd.read_csv(A/'features/sam2_crop.csv').set_index('id').loc[rows.id].reset_index(drop=True)
for family,new,base in [('full_anchor',full,'anchor'),('crop_base',crop,'base13'),('crop_anchor',crop,'anchor')]:
    for seed in [5601,5602,5603]:
        name=f'gpu_sam2_{family}_s{seed}';pred=[]
        for fit in DAYS:
            tr=rows.split.eq('oof')&rows.day.eq(fit);ev=~tr
            X=pd.concat([pd.read_csv(A/f'features/{base}_fit{fit[-2:]}_bank.csv'),new],axis=1).fillna(0)
            model=TabICLClassifier(n_estimators=16,model_path=checkpoint,allow_auto_download=False,device='cuda:0',n_jobs=2,batch_size=2,random_state=seed,use_amp=False)
            model.fit(X.loc[tr],rows.loc[tr,'error'].astype(int));prob=model.predict_proba(X.loc[ev])[:,1]
            pred.append(rows.loc[ev,['id','split']].assign(fit_day=fit,p_error=prob))
            model.save(A/f'models/{name}_fit{fit[-2:]}.pkl',save_model_weights=False,save_training_data=True,save_kv_cache=False)
        d=pd.concat(pred);d.to_csv(A/f'features/{name}_predictions.csv',index=False);print('SAM2_CONFIRMED',name,flush=True)
print('TABULAR_DONE',flush=True)
