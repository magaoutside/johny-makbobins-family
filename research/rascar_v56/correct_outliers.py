"""Automatic correction study. Never uses visual test flags for fitting/selection.

Lightweight numpy/pandas only. Expert probabilities were generated with day-held-out
models; any expert training is done on Kaggle. Hyperparameters are compared on the
existing OOF set, so the winning OOF is a selection estimate, not a new holdout.
"""
from pathlib import Path
import json, hashlib, itertools, os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP

A=Path(__file__).resolve().parent
SEED=os.environ.get('RASCAR_CORRECTION_SEED')
OUT=A/'correction_audit'
if SEED: OUT=OUT/f'seed_{SEED}'
OUT.mkdir(exist_ok=True,parents=True)
Q=pd.read_csv(A/'kaggle_payload/rascar_v56/inputs/q.csv')
T=pd.read_csv(A/'kaggle_payload/rascar_v56/inputs/t.csv')
Y=Q.error.to_numpy(); BO=Q.v52.to_numpy(); BT=T.v52.to_numpy()
DAYS=sorted(Q.day.unique()); CLASSES=sorted(Q.class_name.unique())
all_results=[]; scores={}

def metric(s):
    d={'ap':float(AP(Y,s))}
    for col in ['day','class_name']:
        for k in sorted(Q[col].unique()):
            ix=Q[col].eq(k).to_numpy(); d[str(k)]=float(AP(Y[ix],s[ix]))
    err=Y==1; correct=Y==0
    pair=s[err,None] <= s[None,correct]
    same=Q.class_name.to_numpy()[err,None]==Q.class_name.to_numpy()[None,correct]
    d.update(wrong_pairs=int(pair.sum()),within_class_pairs=int((pair&same).sum()),cross_class_pairs=int((pair&~same).sum()))
    return d

BASE=metric(BO)
def add(name,s,z,**kw):
    s=np.clip(s,0,1); z=np.clip(z,0,1)
    assert np.isfinite(s).all() and np.isfinite(z).all()
    m=metric(s); m.update(name=name,delta=m['ap']-BASE['ap'],**kw)
    m['min_day_delta']=min(m[d]-BASE[d] for d in DAYS)
    m['max_test_delta']=float(abs(z-BT).max())
    scores[name]=(s,z); all_results.append(m)

def expert(name):
    remote=A/'gpu_recovered/rascar_v56/features'
    files=sorted(remote.glob(f'gpu_sam2_{name}_s*_predictions.csv')) if remote.exists() else []
    if SEED: files=[p for p in files if f'_s{SEED}_' in p.name]
    if files:
        raw=pd.concat([pd.read_csv(p) for p in files]).groupby(['id','split','fit_day'],as_index=False).p_error.mean()
    else:
        raw=pd.read_csv(A/f'features/sam2_{name}_icl1_predictions.csv')
    assert not raw[raw.split.eq('oof')].id.duplicated().any()
    o=raw[raw.split.eq('oof')].set_index('id').loc[Q.id,'p_error'].to_numpy()
    t=raw[raw.split.eq('test')].groupby('id').p_error.mean().loc[T.id].to_numpy()
    return o,t,len(files) or 1

def adapt_prior(raw,strength):
    """Unsupervised label-shift EM. Only fit-day labels provide the source prior.

    The target batch contributes predictions and day/class, never error labels.
    Applying the same transductive procedure to held-out days avoids test-only
    tuning. Shrinkage prevents tiny target batches from forcing priors to 0/1.
    """
    rows=pd.concat([Q.assign(split='oof'),T.assign(split='test')],ignore_index=True).set_index('id')
    d=raw.merge(rows[['day','class_name']],left_on='id',right_index=True,validate='many_to_one').copy()
    d['adapted']=np.nan;prior_rows=[]
    for (fit,split,day,cls),g in d.groupby(['fit_day','split','day','class_name']):
        train=Q[Q.day.eq(fit)&Q.class_name.eq(cls)]
        prior=(train.error.sum()+1)/(len(train)+2)
        target=prior;p=np.clip(g.p_error.to_numpy(),.0001,.9999)
        for _ in range(100):
            post=sigmoid(logit(p)+logit(target)-logit(prior))
            nxt=(post.sum()+strength*prior)/(len(post)+strength)
            if abs(nxt-target)<1e-8:break
            target=nxt
        d.loc[g.index,'adapted']=sigmoid(logit(p)+logit(target)-logit(prior))
        prior_rows.append(dict(fit_day=fit,split=split,target_day=day,class_name=cls,target_count=len(g),source_prior=float(prior),estimated_target_prior=float(target),strength=strength))
    pd.DataFrame(prior_rows).to_csv(OUT/f'prior_em_k{strength}_priors.csv',index=False)
    o=d[d.split.eq('oof')].set_index('id').loc[Q.id,'adapted'].to_numpy()
    t=d[d.split.eq('test')].groupby('id').adapted.mean().loc[T.id].to_numpy()
    assert np.isfinite(o).all() and np.isfinite(t).all()
    return o,t

def prior_offsets(rows, train):
    # Only class/labels from training day. No OOF score fitting and no test labels.
    global_p=(train.error.sum()+2)/(len(train)+4)
    cnt=train.groupby('class_name').error.agg(['sum','count'])
    p=(cnt['sum']+12*global_p)/(cnt['count']+12)
    logit=lambda x: np.log(x/(1-x))
    return rows.class_name.map(logit(p)-logit(global_p)).to_numpy()

def sigmoid(x): return 1/(1+np.exp(-x))
def logit(x): x=np.clip(x,.0001,.9999); return np.log(x/(1-x))

def duplicate_pressure(rows):
    pressure=np.zeros(len(rows)); coords=rows[['x1','y1','x2','y2']].to_numpy()
    for _,group in rows.groupby(['frame_id','class_name']):
        ix=group.index.to_numpy(); b=coords[ix]
        if len(ix)<2: continue
        inter=np.maximum(0,np.minimum(b[:,None,2:],b[None,:,2:])-np.maximum(b[:,None,:2],b[None,:,:2])).prod(2)
        area=(b[:,2:]-b[:,:2]).prod(1)
        iou=inter/np.maximum(area[:,None]+area[None,:]-inter,1e-8)
        conf=rows.loc[ix,'confidence'].to_numpy()
        stronger=(conf[None,:]>conf[:,None]+.01)
        pressure[ix]=np.where(stronger,np.clip((iou-.6)/.4,0,1),0).max(1)
    return pressure

def run():
    add('v52',BO,BT,kind='baseline')
    a40o=pd.read_csv(A/'results/oof/v1_anchor_mean3_nude_hands_a40.csv').set_index('id').loc[Q.id,'score'].to_numpy()
    a40t=pd.read_csv(A/'release/submission_v56_tabicl_hands_a40.csv').set_index('id').loc[T.id,'score'].to_numpy()
    add('a40',a40o,a40t,kind='baseline')
    co=np.zeros(len(Q))
    for day in DAYS:
        ix=Q.day.eq(day); co[ix]=prior_offsets(Q.loc[ix],Q[~ix])
    ct=np.mean([prior_offsets(T,Q[Q.day.eq(day)]) for day in DAYS],axis=0)
    for tau in [.05,.10,.20]:
        add(f'class_prior_t{tau:g}',sigmoid(logit(BO)+tau*co),sigmoid(logit(BT)+tau*ct),kind='class_prior',tau=tau)
    # Training audit shows many correct cups/trays above actual errors of other
    # classes. These global monotone transforms preserve ALL within-class ranks.
    # Values are validation-tuned, explicitly not fitted against visual test flags.
    for scope in ['cup_tray','cup','tray']:
        classes=['cup','tray'] if scope=='cup_tray' else [scope]
        om=Q.class_name.isin(classes).to_numpy();tm=T.class_name.isin(classes).to_numpy()
        for shift in [.2,.4,.6]:
            s=BO.copy();z=BT.copy()
            s[om]=sigmoid(logit(BO[om])-shift);z[tm]=sigmoid(logit(BT[tm])-shift)
            add(f'class_odds_{scope}_m{shift:g}',s,z,kind='class_odds',scope=scope,shift=shift)
    do,dt=duplicate_pressure(Q),duplicate_pressure(T)
    for strength in [.05,.10]:
        add(f'duplicate_d{strength:g}',BO+strength*do,BT+strength*dt,kind='duplicate',strength=strength)
    ex={name:expert(name) for name in ['full_anchor','crop_anchor','crop_base']}
    anchor_files=sorted((A/'features').glob('tabiclv1_anchor_n*_s*_predictions.csv'))
    if SEED: anchor_files=[p for p in anchor_files if f'_s{SEED}_' in p.name]
    anchor=pd.concat([pd.read_csv(p) for p in anchor_files]).groupby(['id','split','fit_day'],as_index=False).p_error.mean()
    for strength in [10,30]:
        o,t=adapt_prior(anchor,strength)
        for scope in ['all','nude_hands']:
            om=np.ones(len(Q),bool) if scope=='all' else Q.class_name.eq(scope).to_numpy()
            tm=np.ones(len(T),bool) if scope=='all' else T.class_name.eq(scope).to_numpy()
            for cap in [.05,.10]:
                s=BO.copy();z=BT.copy()
                s[om]+=np.clip(.2*(o-BO),-cap,cap)[om];z[tm]+=np.clip(.2*(t-BT),-cap,cap)[tm]
                add(f'prior_em_k{strength}_{scope}_cap{cap:g}',s,z,kind='prior_em',strength=strength,scope=scope,cap=cap)
    for name,(o,t,n) in ex.items():
        for scope in ['all','person','cup','nude_hands']:
            om=np.ones(len(Q),bool) if scope=='all' else Q.class_name.eq(scope).to_numpy()
            tm=np.ones(len(T),bool) if scope=='all' else T.class_name.eq(scope).to_numpy()
            for cap in [.05,.10]:
                s=BO.copy(); z=BT.copy()
                s[om]+=np.clip(.2*(o-BO),-cap,cap)[om]
                z[tm]+=np.clip(.2*(t-BT),-cap,cap)[tm]
                add(f'{name}_{scope}_cap{cap:g}',s,z,kind='sam2_bounded',family=name,scope=scope,cap=cap,seeds=n)
    eo=np.stack([x[0] for x in ex.values()]); et=np.stack([x[1] for x in ex.values()])
    # Require all three feature families to support the direction; shrink by the
    # least confident family. This has no per-ID hand-coded exceptions.
    def agreement(pred,base,margin,cap):
        d=pred-base[None,:]
        common=np.where(d.min(0)>margin,d.min(0),np.where(d.max(0)<-margin,d.max(0),0))
        return base+np.clip(.25*common,-cap,cap)
    for margin,cap in itertools.product([0,.15],[.05,.10]):
        add(f'sam_consensus_m{margin:g}_cap{cap:g}',agreement(eo,BO,margin,cap),agreement(et,BT,margin,cap),kind='sam_consensus',margin=margin,cap=cap)
    # Majority agreement is less brittle than unanimity. Confident predictions
    # must cross a fixed 0.25/0.75 threshold; cap limits a mistaken correction.
    for cap in [.05,.10]:
        def majority(pred,base):
            med=np.median(pred,axis=0)
            d=np.where(((med<.25)&(med<base))|((med>.75)&(med>base)),.2*(med-base),0)
            return base+np.clip(d,-cap,cap)
        add(f'sam_majority_cap{cap:g}',majority(eo,BO),majority(et,BT),kind='sam_majority',cap=cap)
    # Opposite effects need not be equally reliable. Evaluate one-sided
    # corrections as well; these rules still apply to every box automatically.
    for direction in ['down','up']:
        for family in ['full_anchor','crop_anchor']:
            o,t,n=ex[family]
            for scope in ['all','nude_hands']:
                om=np.ones(len(Q),bool) if scope=='all' else Q.class_name.eq(scope).to_numpy()
                tm=np.ones(len(T),bool) if scope=='all' else T.class_name.eq(scope).to_numpy()
                lo,hi=(-.05,0) if direction=='down' else (0,.05)
                s=BO.copy();z=BT.copy()
                s[om]+=np.clip(.2*(o-BO),lo,hi)[om];z[tm]+=np.clip(.2*(t-BT),lo,hi)[tm]
                add(f'{family}_{scope}_{direction}',s,z,kind='sam_directional',scope=scope,direction=direction,family=family)
    # Segmentation geometry alone is an intentionally simple control.
    for mode in ['full','crop']:
        geom=pd.read_csv(A/f'features/sam2_{mode}.csv').set_index('id')
        def geometry(rows,base):
            x=geom.loc[rows.id];prefix=f'sam2_{mode}_'
            stable=(x[prefix+'confidence_mean'].to_numpy()>.85)&(x[prefix+'bbox_iou_std'].to_numpy()<.05)
            target=np.clip((.8-x[prefix+'bbox_iou'].to_numpy())/.4,0,1)
            return base+stable*np.clip(.2*(target-base),-.05,.05)
        add(f'sam_geometry_{mode}',geometry(Q,BO),geometry(T,BT),kind='sam_geometry')
    # Context expert can discover omissions, whereas crop expert can verify an
    # object. Combine the two directional residuals; conflicting evidence cancels.
    for strength,cap in itertools.product([.1,.2],[.05,.10]):
        def dual(full,crop,base):
            return base+np.clip(strength*(full-base),0,cap)+np.clip(strength*(crop-base),-cap,0)
        add(f'dual_direction_s{strength:g}_cap{cap:g}',dual(ex['full_anchor'][0],ex['crop_anchor'][0],BO),dual(ex['full_anchor'][1],ex['crop_anchor'][1],BT),kind='dual_direction',strength=strength,cap=cap)
    dual_o,dual_t=scores['dual_direction_s0.2_cap0.1']
    for shift in [.2,.4]:
        om=Q.class_name.eq('tray').to_numpy();tm=T.class_name.eq('tray').to_numpy()
        s=dual_o.copy();z=dual_t.copy()
        s[om]=sigmoid(logit(s[om])-shift);z[tm]=sigmoid(logit(z[tm])-shift)
        add(f'dual_plus_tray_m{shift:g}',s,z,kind='dual_plus_tray',shift=shift)
    for strength,cap in itertools.product([10,30],[.05,.10]):
        for tray_shift in [0,.2]:
            source='dual_direction_s0.2_cap0.1' if not tray_shift else 'dual_plus_tray_m0.2'
            s,z=[x.copy() for x in scores[source]]
            eo,et=scores[f'prior_em_k{strength}_nude_hands_cap{cap:g}']
            om=Q.class_name.eq('nude_hands').to_numpy();tm=T.class_name.eq('nude_hands').to_numpy()
            s[om]=eo[om];z[tm]=et[tm]
            add(f'dual_emhands_k{strength}_cap{cap:g}_tray{tray_shift:g}',s,z,kind='dual_emhands',em_strength=strength,em_cap=cap,tray_shift=tray_shift)
    # Regression check: baseline tray within-class AP is already 1.0. Preserve
    # its order exactly and adjust only its scale, avoiding new tray inversions.
    for shift in [0,.2,.4]:
        s,z=[x.copy() for x in scores['dual_emhands_k10_cap0.1_tray0']]
        om=Q.class_name.eq('tray').to_numpy();tm=T.class_name.eq('tray').to_numpy()
        s[om]=sigmoid(logit(BO[om])-shift);z[tm]=sigmoid(logit(BT[tm])-shift)
        add(f'dual_emhands_monotone_tray_m{shift:g}',s,z,kind='monotone_tray',shift=shift)
    remote=A/'gpu_recovered/rascar_v56'
    for path in sorted((remote/'results/oof').glob('vit_*.csv')):
        test_path=remote/'results/submissions'/path.name
        if not test_path.exists():continue
        o=pd.read_csv(path).set_index('id').loc[Q.id,'score'].to_numpy()
        t=pd.read_csv(test_path).set_index('id').loc[T.id,'score'].to_numpy()
        add(path.stem,o,t,kind='dense_gpu')
    # A fixed small combination study, only if an independent backbone candidate
    # improves both held-out days on its own.
    eligible_dense=[r for r in all_results if r['kind']=='dense_gpu' and r['min_day_delta']>=0 and r['delta']>0]
    if eligible_dense:
        best_dense=max(eligible_dense,key=lambda r:r['ap'])['name']
        oo,tt=scores[best_dense]
        for weight in [.5,1.]:
            add(f'dual_plus_dense_w{weight:g}',BO+np.clip(dual_o-BO+weight*(oo-BO),-.15,.15),BT+np.clip(dual_t-BT+weight*(tt-BT),-.15,.15),kind='dual_plus_dense',dense_source=best_dense,weight=weight)
    result=pd.DataFrame(all_results).sort_values('ap',ascending=False)
    # Selection deliberately completed BEFORE reading any audit flags.
    eligible=result[(result.kind!='baseline')&(result.min_day_delta>=-1e-12)&(result.delta>0)]
    # Seed stability is another validation-only filter, independent of test flags.
    if not SEED:
        tables=[OUT/f'seed_{seed}/validation_all.csv' for seed in [5601,5602,5603]]
        if all(p.exists() for p in tables):
            stable=pd.concat([pd.read_csv(p)[['name','min_day_delta']] for p in tables]).groupby('name').min_day_delta.min()
            result['worst_seed_day_delta']=result.name.map(stable)
            eligible=eligible[eligible.name.map(stable).fillna(-1)>=-1e-12]
    result['eligible']=result.name.isin(eligible.name)
    result.to_csv(OUT/'validation_all.csv',index=False)
    if len(eligible):
        winner=eligible.sort_values(['ap','min_day_delta'],ascending=False).iloc[0]['name']
    else: winner='v52'
    s,z=scores[winner]
    Q.assign(score=s,delta=s-BO).to_csv(OUT/'selected_oof.csv',index=False)
    T[['id']].assign(score=z).to_csv(OUT/'submission_v57_auto_correction.csv',index=False)
    np.savez_compressed(OUT/'all_scores.npz',**{f'{k}_{split}':v[i] for k,v in scores.items() for i,split in enumerate(['oof','test'])})
    (OUT/'selection.json').write_text(json.dumps({'winner':winner,'selection':'Highest existing OOF AP among methods non-worse on both days and all three seed/day checks when available; chosen before reading visual test flags. This is validation selection, not a fresh independent holdout.','base':BASE,'selected':metric(s),'tested':len(result)-2,'expert_seeds':{k:v[2] for k,v in ex.items()}},indent=2))
    # Diagnostic only: compare chosen scores to previously saved visual hypotheses.
    audit=pd.read_csv(A/'visual_audit/test_all_362.csv').set_index('id').loc[T.id].reset_index()
    audit['score_v57']=z; audit['delta_vs_v52']=z-BT; audit['delta_vs_a40']=z-a40t
    audit['expected_sign']=audit.flag.map({'up':1,'down':-1}).fillna(0)
    audit['directional_change_vs_v52']=audit.expected_sign*audit.delta_vs_v52
    audit['directional_change_vs_a40']=audit.expected_sign*audit.delta_vs_a40
    flagged=audit[audit.expected_sign.ne(0)].copy()
    for ref in ['v52','a40']:
        d=flagged[f'directional_change_vs_{ref}']
        flagged[f'outcome_vs_{ref}']=np.where(d>=.01,'favorable',np.where(d<=-.01,'unfavorable','under_0.01'))
    audit.to_csv(OUT/'all_test_changes.csv',index=False)
    flagged.to_csv(OUT/'all_43_suspected_changes.csv',index=False)
    counts={}
    for ref in ['v52','a40']:
        counts[ref]=flagged[f'outcome_vs_{ref}'].value_counts().to_dict()
        counts[ref+'_frames']=flagged[flagged[f'outcome_vs_{ref}'].eq('favorable')].frame_id.nunique()
    (OUT/'visual_counts.json').write_text(json.dumps(counts,indent=2))
    print(result[['name','ap',*DAYS,'wrong_pairs','min_day_delta']].head(18).to_string(index=False))
    print('SELECTED',winner,'COUNTS',counts)
    print('CLASS_PRIORS',Q.groupby(['day','class_name']).error.agg(['sum','count','mean']).to_string())

if __name__=='__main__': run()
