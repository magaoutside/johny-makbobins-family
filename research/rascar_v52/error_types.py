"""Separate supervision for correct / low-GT-overlap / localization errors."""
from train import *
def main():
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning);rows,banks,groups,_=load_bank();q,t,h=inputs();bo,bt=baseline(q,t);target={};audit=[]
    from common import iou
    for _,r in q.iterrows():
        gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy();ov=float(iou(r[B],gt).max()) if len(gt) else 0.;typ=0 if r.error==0 else 2 if ov<.05 else 1;target[r.id]=typ;audit.append(dict(id=r.id,error=int(r.error),max_gt_iou=ov,error_type=typ))
    pd.DataFrame(audit).to_csv(A/'inputs/error_type_targets.csv',index=False);rows['error_type']=rows.id.map(target);out=[];manifest=[]
    for key in ['base13','base23','base23_siglip2','base23_nested','all_frozen_emb_all']:
        gr,emb=configurations()[key];cc=list(dict.fromkeys(c for g in gr+['embed_'+e for e in emb] for c in groups[g]))+['class_name']
        for depth in [2,4]:
            name=f'types_{key}_d{depth}';pp=[]
            for fit in DAYS:
                tr=rows.split.eq('oof')&rows.day.eq(fit);ev=~tr;X=banks[fit][cc];model=CatBoostClassifier(iterations=250 if depth==2 else 400,depth=depth,learning_rate=.03,l2_leaf_reg=10 if depth==2 else 20,loss_function='MultiClass',random_seed=23,thread_count=3,verbose=False,allow_writing_files=False);model.fit(X.loc[tr],rows.loc[tr,'error_type'].astype(int),cat_features=['class_name']);prob=model.predict_proba(X.loc[ev]);p=1-prob[:,list(model.classes_).index(0)];pp.append(rows.loc[ev,['id','frame_id','day','class_name','split']].assign(p_error=p,fit_day=fit));model.save_model(str(A/f'models/{name}_fit{fit[-2:]}.cbm'));manifest.append(dict(model=name,fit_day=fit,train_ids=rows.loc[tr,'id'].tolist(),features=cc,targets='0 correct, 1 error with same-class maxIoU>=.05, 2 error maxIoU<.05; type2 is low-GT-overlap, not a proven semantic absence label',GT_used_only_for_training_targets=True))
            pp=pd.concat(pp);pp.to_csv(A/f'features/{name}_predictions.csv',index=False);oo=pp[pp.split.eq('oof')].set_index('id').loc[q.id].p_error.to_numpy();tt=pp[pp.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
            for a in [.05,.1,.2,.5,1000000.]:
                tag='standalone' if a>100 else f'a{int(a*100):02}';s=oo if a>100 else (bo+a*oo)/(1+a);st=tt if a>100 else (bt+a*tt)/(1+a);exp=name+'_'+tag;q.assign(score=s).to_csv(A/f'results/oof/{exp}.csv',index=False);t[['id']].assign(score=st).to_csv(A/f'results/submissions/{exp}.csv',index=False);out.append(dict(name=exp,model=name,alpha=a,**metrics(q,s)))
            pd.DataFrame(out).to_csv(A/'results/error_type_experiments.csv',index=False);(A/'models/error_type_manifest.json').write_text(json.dumps(manifest,indent=2));print(name,AP(q.error,oo),flush=True)
if __name__=='__main__':main()
