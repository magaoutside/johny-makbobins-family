"""Limited, fixed scoring ablations; preserve old files and all new outcomes."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v33'
sys.path.insert(0,str(ROOT/'rascar_v23'));sys.path.insert(0,str(ROOT/'rascar_v25'))
from common23 import q,te,CL,DAYS,B,base_oof,current,align,metric,iou
from common25 import guard
import numpy as np,pandas as pd
from scipy.stats import qmc

def distribution_risk(e,fit,kind):
    z=np.load(A/f'features/side_{kind}_fit{fit[-2:]}_distributions.npz');lookup={i:k for k,i in enumerate(z['ids'])}
    rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv').set_index('id');draw=qmc.Sobol(4,scramble=True,seed=3303).random_base2(9);out=[]
    for _,r in e.iterrows():
        rr=rec.loc[r.id];b=rr[B].to_numpy(float);cb=rr[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(float)
        query=(b-np.r_[cb[:2],cb[:2]])/np.r_[cb[2:]-cb[:2],cb[2:]-cb[:2]]
        p=z['p'][lookup[r.id]].astype(float);p/=p.sum(1,keepdims=True)
        boxes=np.stack([np.minimum(31,np.searchsorted(np.cumsum(p[k]),draw[:,k])) for k in range(4)],1)/32+.5/32
        valid=(boxes[:,2:]>boxes[:,:2]).all(1)
        # Invalid independent-side combinations contribute error; this is a model
        # risk score, not a calibrated probability of official failure.
        risk=np.ones(len(boxes));risk[valid]=iou(query,boxes[valid]).ravel()<.6
        out.append(float(risk.mean()))
    return np.array(out)

def signal(e,name):
    if name=='boundary_conservative':return np.clip((.85-e.optimistic_iou.to_numpy())/.5,0,1)
    if name=='learned_quality':
        return 1-e.predicted_quality.to_numpy() if e.attrs.get('regional') else np.clip((.85-e.predicted_quality.to_numpy())/.5,0,1)
    if name=='mode_iou':return np.clip((.85-e.ref_iou.to_numpy())/.5,0,1)
    if name=='absence':return 1-e.presence.to_numpy()
    if name=='side_distribution':return 1-e.presence.to_numpy()*(1-e.side_risk.to_numpy())
    raise ValueError(name)

def main():
    parents={'v22':(base_oof,current),'v32':(
        align(ROOT/'rascar_v32/results/oof/compose_watershed1_confirmed_hand_protection2.csv',q,'score'),
        align(ROOT/'rascar_v32/submission_v32_research_oof098800.csv',te,'score'))}
    entries=[];controls=[];audit=align(ROOT/'rascar_v29/audit/all_362_fresh_audit.csv',te)
    bad=(audit.visual_status_v29.isin(['clear_error','likely_error'])|audit.geometry_status.eq('robust_error_under_assumptions')).to_numpy()
    good=audit.visual_status_v29.eq('plausible_correct').to_numpy()
    guards={}
    for split,rows in [('oof',q),('test',te)]:
        g=np.ones(len(rows),bool)
        for cls in CL:
            ids=np.flatnonzero(rows.class_name.eq(cls).to_numpy());g[ids]=guard(split,rows.iloc[ids],cls)
        guards[split]=g
    kinds=['manual','hardneg','regional_hardneg','pool_regional','pool_regional_balanced','pool_regional_balanced_rare']
    for kind in kinds:
        oo=[];tt=[]
        for fit in DAYS:
            allrows=pd.read_csv(A/f'features/side_{kind}_fit{fit[-2:]}.csv');e=allrows[allrows.split.isin(['oof','test'])].copy()
            e['side_risk']=distribution_risk(e,fit,kind)
            oo.append(e[e.split.eq('oof')]);tt.append(e[e.split.eq('test')].set_index('id').loc[te.id].reset_index())
            for (split,cls),sub in allrows[allrows.split.isin(['manual','empty'])].groupby(['split','class_name']):
                controls.append(dict(model=kind,fit_day=fit,split=split,class_name=cls,n_crops=len(sub),presence_mean=float(sub.presence.mean()),presence_gt05=float(sub.presence.ge(.5).mean()),localization_mean_iou=float(sub.ref_iou.mean()) if split=='manual' else None,localization_recall06=float(sub.ref_iou.ge(.6).mean()) if split=='manual' else None))
        oo=pd.concat(oo).set_index('id').loc[q.id].reset_index();assert oo.id.is_unique
        oo.attrs['regional']='regional' in kind
        for t in tt:t.attrs['regional']='regional' in kind
        oo.to_csv(A/f'features/{kind}_oof_evidence.csv',index=False)
        for fit,t in zip(DAYS,tt):t.to_csv(A/f'features/{kind}_test_fit{fit[-2:]}_evidence.csv',index=False)
        for parent,(bs,bt) in parents.items():
            for name in ['boundary_conservative','learned_quality','absence','side_distribution','mode_iou']:
                target=signal(oo,name);targ=np.stack([signal(t,name) for t in tt]);delta=target-bs;dt=targ-bt[None,:]
                same=((dt>=0).all(0)|(dt<=0).all(0));tc=np.where((dt>=0).all(0),dt.min(0),dt.max(0));tc[~same]=0
                if name in ['boundary_conservative','absence']:delta=np.maximum(delta,0);tc=np.maximum(tc,0)
                delta[(delta<0)&~guards['oof']]=0;tc[(tc<0)&~guards['test']]=0
                for weight in [.1,.25]:
                    s=np.clip(bs+weight*delta,0,1);t=np.clip(bt+weight*tc,0,1);label=f'{parent}_{kind}_{name}_w{weight}'
                    q.assign(score=s).to_csv(A/f'results/oof/{label}.csv',index=False);te[['id']].assign(score=t).to_csv(A/f'results/submissions/submission_{label}.csv',index=False)
                    changes=t-bt
                    entries.append(dict(experiment=label,parent=parent,model=kind,signal=name,weight=weight,**metric(q,s),delta_ap_vs_parent=metric(q,s)['pooled_ap']-metric(q,bs)['pooled_ap'],changed_test=int((abs(changes)>1e-9).sum()),known_error_down=int((bad&(changes<-.02)).sum()),plausible_correct_up=int((good&(changes>.02)).sum()),known_error_up=int((bad&(changes>.02)).sum()),plausible_correct_down=int((good&(changes<-.02)).sum())))
    out=pd.DataFrame(entries).sort_values('pooled_ap',ascending=False);out.to_csv(A/'results/experiments.csv',index=False);pd.DataFrame(controls).to_csv(A/'results/manual_and_empty_controls.csv',index=False)
    (A/'results/protocol.json').write_text(json.dumps(dict(n_variants=len(out),weights=[.1,.25],models=kinds,primary_metric='sklearn.average_precision_score',day_split='fit25 predict26 and reverse; queries and pseudo data restricted to fit day',test_labels_used=False,leaderboard_upload=False,scope='All four evaluated classes. Test corrections require two fit-day models to agree in sign. Prior semantic/duplicate guard applies to new rescues.',uncertainty='Side distributions are learned, not calibrated confidence intervals.',selection='Exploratory repeated OOF model selection; no untouched holdout.'),indent=2))
    print(out.to_string(index=False));print(pd.DataFrame(controls).to_string(index=False))
if __name__=='__main__':main()
