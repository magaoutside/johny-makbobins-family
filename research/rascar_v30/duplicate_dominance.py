"""Conditional geometric dominance in duplicate groups.

Unlike forced winner rules, a loser is penalized only when another query is
valid throughout the reference tolerance envelope and dominates the loser's
best possible IoU. Both queries must have no second eligible reference. This
does not prove that the reference detector found every true instance.
"""
from pathlib import Path
import sys,itertools
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v30'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,B,iou,align,current,base_oof
from features23 import nms
from bounded_geometry import save

def interval(query,ref,u):
    choices=[]
    for j in range(4):
        lo,hi=ref[j]-u,ref[j]+u
        choices.append(sorted(set([lo,hi]+[float(v) for v in query[[j%2,j%2+2]] if lo<=v<=hi])))
    bb=np.array(list(itertools.product(*choices)));bb=bb[(bb[:,2:]>bb[:,:2]).all(1)]
    ov=iou(query,bb)[0];return float(ov.min()),float(ov.max())

def features(rows,split,u):
    raw=pd.read_csv(ROOT/f'rascar_v15/v8/fused_{"train" if split=="oof" else "test"}_references.csv')
    out=pd.DataFrame(dict(id=rows.id,dominated=False,winner='',confidence=0.,winner_min=0.,loser_max=1.,margin=0.)).set_index('id')
    for (fid,cls),g in rows[rows.class_name.isin(['cup','nude_hands'])].groupby(['frame_id','class_name']):
        if len(g)<2:continue
        rr=nms(raw[raw.frame_id.eq(fid)&raw.class_name.eq(cls)&raw.confidence.ge(.45)],.5)
        if not len(rr):continue
        low=np.zeros((len(g),len(rr)));high=low.copy()
        for i,b in enumerate(g[B].to_numpy(float)):
            for j,ref in enumerate(rr[B].to_numpy(float)):low[i,j],high[i,j]=interval(b,ref,u)
        for i,r in enumerate(g.itertuples()):
            near=np.flatnonzero(high[i]>=.15)
            for j in near:
                if np.any(np.delete(high[i],j)>=.6):continue
                for k,other in enumerate(g.itertuples()):
                    if k==i or np.any(np.delete(high[k],j)>=.6):continue
                    if low[k,j]>=.6 and low[k,j]>high[i,j]+.02:
                        margin=low[k,j]-high[i,j]
                        if margin>out.loc[r.id,'margin']:
                            out.loc[r.id]=[True,other.id,float(rr.iloc[j].confidence),low[k,j],high[i,j],margin]
    return out.reset_index()

def run():
    results=[]
    for u in [1.,2.]:
        fs={}
        for split,rows in [('oof',q),('test',te)]:
            f=features(rows,split,u);f.to_csv(A/f'features/dominance_u{u:g}_{split}.csv',index=False);fs[split]=f
        for conf in [.65,.8]:
            for weight in [.25,.5]:
                outs=[]
                for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
                    f=fs[split];gate=f.dominated.to_numpy(bool)&f.confidence.ge(conf).to_numpy()
                    outs.append(base+weight*np.maximum(.9-base,0)*gate)
                results.append(save(f'dominance_u{u:g}_c{conf}_w{weight}',*outs,dict(type='robust_one_to_one_dominance',
                    tolerance=u,confidence=conf,weight=weight,test_labels_used=False)))
    pd.DataFrame(results).drop(columns='details').to_csv(A/'results/dominance_metrics.csv',index=False)
    print(pd.DataFrame(results).drop(columns='details').to_string(index=False))

if __name__=='__main__':run()
