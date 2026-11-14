from common43 import *
from scipy.special import expit
from sklearn.metrics import average_precision_score as AP
q,te,h=inputs();bo,bt=baseline(q,te);records=pd.read_csv(A/'inputs/records.csv')
rows=[];raw=[]
def save(name,oo,tt,description):
    result=dict(name=name,description=description,**metrics(q,oo))
    for split,d,p,base in [('oof',q,oo,bo),('test',te,tt,bt)]:
        assert len(p)==len(d) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        path=A/f'results/{"oof" if split=="oof" else "submissions"}/{name}.csv'
        (d.assign(score=p) if split=='oof' else d[['id']].assign(score=p)).to_csv(path,index=False)
        result['changed_'+split]=int((abs(p-base)>1e-10).sum())
        order=np.argsort(-base,kind='stable');v=p[order];result['inversions_'+split]=int(np.triu(v[:,None]<v[None,:],1).sum())
    rows.append(result)
save('V31',bo,bt,'Preserved confirmed best baseline Public .96239')
f=pd.read_csv(A/'features/frozen_coco.csv').set_index('id')
for kind in ['confidence','geometry','combined']:
 for cls in ['person','cup','both']:
    values=[]
    for df,base in [(q,bo),(te,bt)]:
        z=f.loc[df.id];p=1-z.coco_probability.to_numpy() if kind=='confidence' else expit((.6-z.ref_iou.to_numpy())/.08)
        if kind=='combined':p=(1-z.coco_probability.to_numpy())*.5+p*.5
        gate=(df.class_name.isin(['person','cup']) if cls=='both' else df.class_name.eq(cls)).to_numpy()&(z.coco_probability.to_numpy()>.3)
        values.append(base+.1*gate*(p-base))
    save(f'frozen_{kind}_{cls}_w010',*values,'Frozen pretrained COCO ROI; confidence gate >.3; exploratory ablation')
for cut in [.8,.9]:
    values=[]
    for df,base in [(q,bo),(te,bt)]:
        z=f.loc[df.id];gate=df.class_name.eq('person').to_numpy()&(z.coco_probability.to_numpy()>.9)&(z.ref_iou.to_numpy()>cut)
        values.append(base+.1*gate*(np.minimum(1-z.coco_probability.to_numpy(),base)-base))
    save(f'frozen_rescue_person_iou{cut}_w010',*values,'Exploratory strict confirmation control; thresholds not independently validated')
for variant in ['clean','pool']:
 path=A/f'results/{variant}_predictions.csv'
 if not path.exists():continue
 pred=pd.read_csv(path)
 if pred.fit_day.nunique()!=2:continue
 oo=pred[pred.source.eq('query')&pred.day.ne(pred.fit_day)].set_index('id').loc[q.id]
 tt=pred[pred.source.eq('test')].groupby('id').mean(numeric_only=True).loc[te.id]
 for kind in ['error','quality','refinement']:
    po=oo.p_error.to_numpy() if kind=='error' else expit((.6-oo['pred_iou' if kind=='quality' else 'ref_iou'].to_numpy())/.08)
    pt=tt.p_error.to_numpy() if kind=='error' else np.mean([expit((.6-g.set_index('id').loc[te.id,'pred_iou' if kind=='quality' else 'ref_iou'].to_numpy())/.08) for _,g in pred[pred.source.eq('test')].groupby('fit_day')],axis=0)
    raw.append(dict(variant=variant,signal=kind,**metrics(q,po)))
    save(f'{variant}_{kind}_standalone',po,pt,'Standalone diagnostic, not promoted')
    for w in [.05,.10]:
        save(f'{variant}_{kind}_w{int(100*w):03d}',bo+w*(po-bo),bt+w*(pt-bt),'Fixed conservative blend with V31; no test annotations')
 # Reference quality is separately evaluated against official GT.
 evidence=[]
 for _,v in oo.reset_index().iterrows():
    original=q[q.id.eq(v.id)].iloc[0];gt=h[h.frame_id.eq(v.frame_id)&h.class_name.eq(v.class_name)][B].to_numpy(float)
    rv=v[['ref_'+x for x in B]].to_numpy(float);qb=original[B].to_numpy(float)
    evidence.append(dict(id=v.id,day=v.day,class_name=v.class_name,error=original.error,query_gt_iou=float(iou(qb,gt).max()) if len(gt) else 0,reference_gt_iou=float(iou(rv,gt).max()) if len(gt) else 0,estimated_iou=v.pred_iou,p_error=v.p_error))
 pd.DataFrame(evidence).to_csv(A/f'audit/{variant}_reference_quality.csv',index=False)
for variant in ['clean','pool']:
 path=A/f'results/local_{variant}_predictions.csv'
 if not path.exists():continue
 pred=pd.read_csv(path)
 if pred.fit_day.nunique()!=2:continue
 oo=pred[pred.source.eq('query')&pred.day.ne(pred.fit_day)].set_index('id').loc[q.id]
 td=[g.set_index('id').loc[te.id] for _,g in pred[pred.source.eq('test')].groupby('fit_day')]
 for cls in ['person','nude_hands','all']:
  for direction in ['two_sided','rescue_only']:
    po=np.clip((.85-oo.ref_iou.to_numpy())/.5,0,1)
    go=np.isfinite(po)&oo.components.eq(1).to_numpy()&oo.mask_confidence.ge(.7).to_numpy()
    pp=np.stack([np.clip((.85-g.ref_iou.to_numpy())/.5,0,1) for g in td]);dd=pp-bt
    gt=np.isfinite(pp).all(0)&np.stack([g.components.eq(1).to_numpy()&g.mask_confidence.ge(.7).to_numpy() for g in td]).all(0)
    gt&=((dd>=0).all(0)|(dd<=0).all(0))
    pt=np.where((dd>=0).all(0),np.min(pp,0),np.max(pp,0))
    if cls!='all':go&=q.class_name.eq(cls).to_numpy();gt&=te.class_name.eq(cls).to_numpy()
    do=np.where(go,po-bo,0);dt=np.where(gt,pt-bt,0)
    if direction=='rescue_only':do=np.minimum(do,0);dt=np.minimum(dt,0)
    save(f'local_{variant}_{cls}_{direction}_w010',bo+.1*do,bt+.1*dt,'Local detector FPN instance mask; single component/confidence>=.7; test both days agree; cautious signal')
pd.DataFrame(rows).to_csv(A/'results/experiments.csv',index=False);pd.DataFrame(raw).to_csv(A/'results/standalone.csv',index=False)
print(pd.DataFrame(rows)[['name','ap','2026-10-25','2026-10-26','changed_test','inversions_test']].to_string(index=False))
