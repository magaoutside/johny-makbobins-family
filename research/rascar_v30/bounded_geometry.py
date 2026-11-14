"""Local postcorrection: an error must persist over a boundary tolerance set.

This is a deterministic sensitivity envelope, NOT a confidence interval for GT.
It only raises error scores, never deletes queries or automatically rescues a
duplicate winner. Query IDs are used solely for joining results.
"""
from pathlib import Path
import sys,json,itertools
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v30'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,h,B,DAYS,iou,current,base_oof,align,metric
from features23 import nms

def upper_iou(query,reference,u):
    choices=[]
    for j in range(4):
        lo,hi=reference[j]-u,reference[j]+u
        choices.append(sorted(set([lo,hi]+[float(v) for v in query[[j%2,j%2+2]] if lo<=v<=hi])))
    boxes=np.array(list(itertools.product(*choices)))
    boxes=boxes[(boxes[:,2:]>boxes[:,:2]).all(1)]
    return float(iou(query,boxes).max(initial=0))

def extract(rows,split):
    raw=pd.read_csv(ROOT/f'rascar_v15/v8/fused_{"train" if split=="oof" else "test"}_references.csv')
    sem=align(ROOT/f'rascar_v16/semantic_both_{split}.csv',rows)
    out=[]
    for r in rows.itertuples():
        d=dict(id=r.id,confidence=0.,raw_iou=1.,semantic=sem.loc[r.id,'p_'+r.class_name])
        for u in [1.,2.,3.,4.]:d[f'upper_{u:g}']=1.
        for c in B:d['ref_'+c]=0.
        rr=nms(raw[raw.frame_id.eq(r.frame_id)&raw.class_name.eq(r.class_name)&raw.confidence.ge(.45)],.5)
        if len(rr):
            b=np.array([getattr(r,c) for c in B]);rb=rr[B].to_numpy(float)
            ov=iou(b,rb)[0];j=ov.argmax();ref=rb[j]
            d.update(confidence=float(rr.iloc[j].confidence),raw_iou=float(ov[j]),
                     **dict(zip(['ref_'+c for c in B],ref)))
            # All plausible refs are retained: failure against one does not
            # override a possible successful match to a different instance.
            for u in [1.,2.,3.,4.]:d[f'upper_{u:g}']=max(upper_iou(b,x,u) for x in rb)
        out.append(d)
    return pd.DataFrame(out)

def save(name,s,t,details):
    assert np.isfinite(s).all() and np.isfinite(t).all()
    q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False)
    te[['id']].assign(score=t).to_csv(A/f'results/submissions/submission_{name}.csv',index=False)
    audit=pd.read_csv(ROOT/'rascar_v29/audit/all_362_fresh_audit.csv').set_index('id').loc[te.id].reset_index()
    audit['new_score']=t;audit['delta']=t-current
    audit[abs(audit.delta)>1e-9].sort_values('review_no').to_csv(A/f'results/reviews/{name}.csv',index=False)
    good=audit.visual_status_v29.eq('plausible_correct').to_numpy()
    bad=(audit.visual_status_v29.eq('clear_error')|audit.geometry_status.eq('robust_error_under_assumptions')).to_numpy()
    row=dict(experiment=name,**metric(q,s),changed_oof=int((abs(s-base_oof)>1e-9).sum()),
        changed_test=int((abs(t-current)>1e-9).sum()),audit_plausible_raised=int((good&(t>current+1e-9)).sum()),
        audit_error_raised=int((bad&(t>current+1e-9)).sum()),public_ap=None,submitted=False,details=details)
    (A/f'results/{name}.json').write_text(json.dumps(row,indent=2))
    return row

def run():
    fs={}
    for split,rows in [('oof',q),('test',te)]:
        path=A/f'features/bounded_{split}.csv'
        if not path.exists():extract(rows,split).to_csv(path,index=False)
        fs[split]=pd.read_csv(path)
    result=[]
    # Small explicit family, all runs retained, no selecting per-ID corrections.
    for cls in ['cup','nude_hands','person']:
        for u in ([2.,3.] if cls!='person' else [4.]):
            for conf in [.65,.8]:
                for weight in [.25,.5]:
                    outs=[]
                    for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
                        f=fs[split]
                        gate=(rows.class_name.eq(cls).to_numpy()&f.confidence.ge(conf).to_numpy()&f.semantic.ge(.8).to_numpy()
                              &f.raw_iou.ge(.1).to_numpy()&f[f'upper_{u:g}'].lt(.6).to_numpy())
                        # Fixed .9 target avoids making this a binary sort.
                        outs.append(base+weight*np.maximum(.9-base,0)*gate)
                    name=f'bounded_{cls}_u{u:g}_c{conf}_w{weight}'
                    result.append(save(name,*outs,dict(type='bounded_geometry',class_name=cls,tolerance_px=u,
                        confidence=conf,weight=weight,target=.9,test_labels_used=False)))
    pd.DataFrame([{k:v for k,v in r.items() if k!='details'} for r in result]).to_csv(A/'results/bounded_metrics.csv',index=False)
    print(pd.DataFrame(result).drop(columns='details').sort_values('pooled_ap',ascending=False).to_string(index=False))

if __name__=='__main__':run()
