"""Equal weight average of three tray-only linear heads; no seed selection."""
from train_tray import *
from common46 import v22base,compose
def main():
    rec=records();feat=np.load(ROOT/'rascar_v31/features/dense_grids.npy',mmap_mode='r');pool=pd.read_csv(A/'inputs/pool_records.csv');pf=np.load(A/'features/small_pool.npy',mmap_mode='r');truth=pd.read_csv(A/'pool_review/boxes.csv').set_index('frame_id');rows=[];poolrows=[]
    for fit in DAYS:
        states=[torch.load(A/f'models/small_silver_fit{fit[-2:]}{s}.pt',weights_only=True) for s in ['', '_s4902','_s4903']];state={k:torch.stack([s[k] for s in states]).mean(0) for k in states[0]};model=Tray(384);model.load_state_dict(state);model.eval();torch.save(state,A/f'models/small_silver_ensemble_fit{fit[-2:]}.pt')
        ev=rec[rec.class_name.eq('tray')&(rec.split.eq('test')|(rec.split.eq('oof')&rec.day.ne(fit)))];held=pool[pool.day.ne(fit)]
        for split,data,features in [('query_test',ev,feat),('pool',held,pf)]:
            for i,r in data.iterrows():
                with torch.inference_mode():prob=model(torch.tensor(np.array(features[i:i+1],np.float32)))[0].sigmoid().numpy()
                p=np.zeros((4,32,32),np.float32);p[3]=prob;d=decode(p,r)
                if split=='query_test':rows.append(dict(id=r.id,frame_id=r.frame_id,day=r.day,fit_day=fit,split=r.split,source='query' if r.split=='oof' else 'test',class_name='tray',**d))
                else:
                    b=truth.loc[r.frame_id,B].to_numpy(float);ref=np.array([d['ref_'+v] for v in B]);lo,hi=bounds(r[B].to_numpy(float),b,r.edge_uncertainty_px);poolrows.append(dict(review_index=r.review_index,frame_id=r.frame_id,day=r.day,role=r.role,key='small',kind='silver_ensemble',fit_day=fit,reference_gt_iou=float(iou(ref,b)[0,0]) if np.isfinite(ref).all() else 0.,query_iou_low=lo,query_iou_high=hi,robust_label=0 if lo>=.6 else (1 if hi<.6 else np.nan),**d))
    f=pd.DataFrame(rows);f.to_csv(A/'features/ensemble_evidence.csv',index=False);pd.DataFrame(poolrows).to_csv(A/'audit/ensemble_pool_geometry.csv',index=False);q,te,h=inputs();bo,bt=baseline(q,te);bq,bte=v22base(q,te);o=f[f.split.eq('oof')].set_index('id').reindex(q.id);ts=[g.set_index('id').reindex(te.id) for _,g in f[f.split.eq('test')].groupby('fit_day')];os=compose(bq,o.overlap.to_numpy());tt=compose(bte,np.stack([g.overlap for g in ts]));out=[]
    for weight in [.25,.5,1.]:
        s=bo.copy();t=bt.copy();om=q.class_name.eq('tray');tm=te.class_name.eq('tray');s[om]+=weight*(os[om]-bo[om]);t[tm]+=weight*(tt[tm]-bt[tm]);name=f'small_silver_ensemble_tray_mix{int(weight*100)}';q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=t).to_csv(A/f'results/submissions/{name}.csv',index=False);out.append(dict(name=name,**metrics(q,s),changed_test=int((abs(t-bt)>1e-9).sum())))
    pd.DataFrame(out).to_csv(A/'results/ensemble_experiments.csv',index=False);print(pd.DataFrame(out).to_string(index=False))
if __name__=='__main__':
    torch.set_num_threads(3);cv2.setNumThreads(1);main()
