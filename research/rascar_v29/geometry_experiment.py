"""Cup localization uncertainty calibrated on the opposite day's annotations.

No test review labels, test reference boxes drawn by the reviewer, or leaderboard
labels enter this experiment. SAM is frozen. Empirical residual scenarios preserve
the joint relationship between all four coordinates. Duplicates are assigned as a
group in each scenario; one model's multiple views are not independent votes.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v29'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,h,B,DAYS,iou,current,base_oof,align,metric
for folder in ['features','models','results/oof','results/submissions']:(A/folder).mkdir(exist_ok=True,parents=True)

def match(ov,threshold=.6):
    n,m=ov.shape
    if not n or not m:return {}
    bonus=min(n,m)+1;reward=np.zeros((n,m+n))
    reward[:,:m]=np.where(ov>=threshold,bonus+ov,-bonus)
    ii,jj=linear_sum_assignment(-reward)
    return {i:j for i,j in zip(ii,jj) if j<m and ov[i,j]>=threshold}

def references(split):
    raw=pd.read_csv(ROOT/f'rascar_v15/v8/fused_{split}_references.csv')
    raw=raw[raw.class_name.eq('cup')&raw.confidence.ge(.5)]
    groups={}
    for fid,g in raw.groupby('frame_id'):
        keep=[]
        for ix,r in g.sort_values('confidence',ascending=False).iterrows():
            if not keep or iou(r[B],g.loc[keep,B]).max()<.5:keep.append(ix)
        groups[fid]=g.loc[keep].reset_index(drop=True)
    return groups

def calibrate(day,refs):
    rows=[]
    for fid,g in h[h.day.eq(day)&h.class_name.eq('cup')].groupby('frame_id'):
        rr=refs.get(fid)
        if rr is None:continue
        pairs=match(iou(rr[B],g[B]),threshold=.3)
        for ri,gi in pairs.items():
            b=rr.iloc[ri][B].to_numpy(float);gt=g.iloc[gi][B].to_numpy(float)
            delta=(gt-b)/np.tile(np.maximum(b[2:]-b[:2],1),2)
            rows.append(dict(frame_id=fid,fit_day=day,reference_iou=float(iou(b,gt)[0,0]),
                             **dict(zip(B,delta))))
    d=pd.DataFrame(rows);assert len(d)>=8
    d['weight']=1/d.groupby('frame_id').frame_id.transform('size')
    d['weight']/=d.weight.sum()
    d.to_csv(A/f'models/cup_residuals_{day}.csv',index=False)
    return d

def predict(rows,refs,cal,grouped=True):
    out=[]
    for fid,g in rows.groupby('frame_id'):
        g=g[g.class_name.eq('cup')]
        if not len(g):continue
        rr=refs.get(fid)
        if rr is None or not len(rr):
            for r in g.itertuples():out.append(dict(id=r.id,p_error=0.,reference_confidence=0.,raw_iou=0.,scenarios=len(cal)))
            continue
        rb=rr[B].to_numpy(float);ov=iou(g[B],rb);j=ov.argmax(1)
        wh=np.maximum(rb[:,2:]-rb[:,:2],1);probs=np.zeros(len(g))
        for row in cal.itertuples():
            residual=np.array([getattr(row,c) for c in B]);bb=rb+residual*np.tile(wh,(1,2))
            bb=np.clip(bb,0,[1280,720,1280,720]);valid=(bb[:,2:]>bb[:,:2]).all(1)
            over=iou(g[B],bb[valid])
            if grouped:
                assignment=match(over);errors=np.array([k not in assignment for k in range(len(g))],float)
            else:errors=(over.max(1,initial=0)<.6).astype(float)
            probs+=row.weight*errors
        for k,r in enumerate(g.itertuples()):
            out.append(dict(id=r.id,p_error=float(probs[k]),reference_confidence=float(rr.iloc[j[k]].confidence),
                            raw_iou=float(ov[k,j[k]]),scenarios=len(cal)))
    d=pd.DataFrame(out).set_index('id').reindex(rows.id).fillna(0).reset_index()
    return d

def run():
    specification=dict(reference_model='frozen pretrained SAM, highest confidence after NMS IoU .5',
        calibration='GT minus reference, normalized four coordinates, frame-weighted empirical distribution',
        fitting_days=DAYS,held_day_rule='fit25 evaluate26; fit26 evaluate25',
        test_label_use=False,weights=[.1,.25,.5],directions=['penalty','both'],
        group_modes=['independent','one_to_one'],gate='cup, semantic>=.8, reference confidence>=.65, raw IoU>=.2',
        purpose='check whether geometry-aware uncertainty is safer than V27 class-confirmation rescue')
    (A/'inputs/geometry_experiment_spec.json').write_text(json.dumps(specification,indent=2))
    trainrefs=references('train');testrefs=references('test')
    cals={day:calibrate(day,trainrefs) for day in DAYS};metrics=[]
    for grouped in [False,True]:
        mode='one_to_one' if grouped else 'independent'
        fo=pd.DataFrame({'id':q.id,'p_error':0.,'reference_confidence':0.,'raw_iou':0.})
        tests=[]
        for fit in DAYS:
            held=q[q.day.ne(fit)]
            assert set(cals[fit].frame_id).isdisjoint(held.frame_id)
            pr=predict(held,trainrefs,cals[fit],grouped).set_index('id')
            for c in ['p_error','reference_confidence','raw_iou']:fo.loc[q.day.ne(fit),c]=pr[c].reindex(held.id).to_numpy()
            tests.append(predict(te,testrefs,cals[fit],grouped))
        ft=tests[0].copy();ft['p_error']=(tests[0].p_error+tests[1].p_error)/2
        ft['fold_min']=np.minimum(tests[0].p_error,tests[1].p_error);ft['fold_max']=np.maximum(tests[0].p_error,tests[1].p_error)
        fo.to_csv(A/f'features/{mode}_oof.csv',index=False);ft.to_csv(A/f'features/{mode}_test.csv',index=False)
        for direction in ['penalty','both']:
            for weight in [.1,.25,.5]:
                outputs=[]
                for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
                    sem=align(ROOT/f'rascar_v16/semantic_both_{split}.csv',rows,'p_cup')
                    gate=rows.class_name.eq('cup').to_numpy()&(sem>=.8)&f.reference_confidence.ge(.65).to_numpy()&f.raw_iou.ge(.2).to_numpy()
                    delta=f.p_error.to_numpy()-base
                    if direction=='penalty':
                        gate&=f.p_error.ge(.65).to_numpy()
                        if split=='test':gate&=f.fold_min.ge(.65).to_numpy()
                        delta=np.maximum(delta,0)
                    s=np.clip(base+weight*delta*gate,0,1);outputs.append(s)
                name=f'cup_uncertainty_{mode}_{direction}_w{weight}';s,t=outputs
                q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False)
                te[['id']].assign(score=t).to_csv(A/f'results/submissions/submission_{name}.csv',index=False)
                metrics.append(dict(experiment=name,**metric(q,s),changed_oof=int((abs(s-base_oof)>1e-9).sum()),
                    changed_test=int((abs(t-current)>1e-9).sum()),public_ap=None,submitted=False))
    metrics=pd.DataFrame(metrics);metrics.to_csv(A/'results/geometry_metrics.csv',index=False)
    print('CALIBRATION',[(d,len(c),c.frame_id.nunique()) for d,c in cals.items()])
    print('BASE',metric(q,base_oof));print(metrics.to_string(index=False))

if __name__=='__main__':run()
