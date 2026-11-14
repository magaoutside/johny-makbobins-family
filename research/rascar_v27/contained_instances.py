"""Detect merged people through contained, distinct reference cores.

Two head centres alone are insufficient: a head can enter a correct neighbour's
box. Each complete reference must be mostly contained, and neither may explain
the query with IoU >= .6. Uses frozen pretrained detections, no test labels.
"""
from common27 import *
from itertools import combinations

def features(rows,split):
    refs=pd.read_csv(ROOT/f'rascar_v15/v8/fused_{"train" if split=="oof" else "test"}_references.csv');out=[]
    for r in rows.itertuples():
        f=dict(id=r.id,merged=False,contained=0,best_iou=0.,pair_iou=1.,min_conf=0.,core_separation=0.)
        if r.class_name!='person':out.append(f);continue
        b=np.array([getattr(r,c) for c in B]);g=refs[refs.frame_id.eq(r.frame_id)&refs.class_name.eq('person')&refs.confidence.ge(.75)]
        if len(g)<2:out.append(f);continue
        rb=g[B].to_numpy();wh=rb[:,2:]-rb[:,:2];inter=np.maximum(0,np.minimum(b[2:],rb[:,2:])-np.maximum(b[:2],rb[:,:2])).prod(1)
        coverage=inter/wh.prod(1);overlap=iou(b,rb)[0];inside=np.flatnonzero(coverage>=.9)
        f.update(contained=len(inside),best_iou=float(overlap.max()))
        if overlap.max()>=.6:out.append(f);continue
        for a,c in combinations(inside,2):
            pair=float(iou(rb[a],rb[c])[0,0]);
            # Upper-body cores at different horizontal positions distinguish
            # neighbours from nested short/full boxes of one person.
            sep=abs((rb[a,0]+rb[a,2]-rb[c,0]-rb[c,2])/2)/min(wh[a,0],wh[c,0])
            tops=abs(rb[a,1]-rb[c,1])/min(wh[a,1],wh[c,1])
            if pair<=.4 and sep>=.45 and tops<=.35:
                f.update(merged=True,pair_iou=pair,min_conf=float(min(g.iloc[a].confidence,g.iloc[c].confidence)),core_separation=sep)
        out.append(f)
    return pd.DataFrame(out)
if __name__=='__main__':
    fs={s:features(rows,s) for s,rows in [('oof',q),('test',te)]}
    for s,f in fs.items():f.to_csv(A/f'features/contained_people_{s}.csv',index=False)
    for weight in [.3,.6]:
        outputs=[]
        for split,base in [('oof',base_oof),('test',current)]:
            f=fs[split];outputs.append(base+weight*np.maximum(.95-base,0)*f.merged.to_numpy())
        register(f'contained_people_w{weight}',*outputs,dict(type='two_contained_person_references',min_conf=.75,max_individual_iou=.6,min_coverage=.9,max_pair_iou=.4,min_core_separation=.45,weight=weight))
