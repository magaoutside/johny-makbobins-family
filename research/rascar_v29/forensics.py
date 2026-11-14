"""Rank and duplicate diagnostics, not an estimate of hidden leaderboard AP."""
from pathlib import Path
import itertools,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v29'
sys.path.insert(0,str(A));from audit_geometry import interval_iou,iou,B

GROUPS=[([18,19],1),([24,25],1),([39,40],1),([45,46],1),([49,50],1),([71,72],1),
    ([110,111],1),([136,137],1),([145,146,147],1),([160,161],2),([172,173],1),
    ([192,193,194,195],2),([206,207],1),([238,239,240],1),([251,252],1),([260,261],1),
    ([278,279],1),([319,320],1),([325,326],1),([334,335],1)]

def run():
    d=pd.read_csv(A/'audit/all_362_fresh_audit.csv').set_index('review_no')
    other=pd.read_csv(A/'audit/other_geometry.csv').set_index('review_no')
    for no,r in other.iterrows():
        d.loc[no,['manual_max_iou','iou_min','iou_max']]=[r.manual_iou,r.iou_min,r.iou_max]
        d.loc[no,'geometry_status']='robust_error_under_assumptions' if r.iou_max<.6 else 'robust_geometry_under_assumptions' if r.iou_min>=.6 else 'boundary_uncertain'
    d['group_id']='';d['group_visible_objects']=np.nan
    groups=[]
    for k,(nos,objects) in enumerate(GROUPS,1):
        g=d.loc[nos];assert g.frame_id.nunique()==1 and g.class_name.nunique()==1
        ident=f'group_{k:02d}';d.loc[nos,'group_id']=ident;d.loc[nos,'group_visible_objects']=objects
        groups.append(dict(group_id=ident,review_nos=','.join(map(str,nos)),frame_id=g.frame_id.iloc[0],class_name=g.class_name.iloc[0],
            visible_objects=objects,query_count=len(g),minimum_errors_if_object_count_is_right=max(0,len(g)-objects),
            note='One-to-one upper bound, not a declaration that a winner is correct; all boxes can fail IoU.'))
    pd.DataFrame(groups).to_csv(A/'audit/duplicate_groups.csv',index=False)
    cups=pd.read_csv(A/'annotations/cup_tight_boxes.csv');rest=pd.read_csv(A/'annotations/other_tight_boxes.csv')
    estimates=pd.concat([cups,rest]);stability=[]
    for k,(nos,objects) in enumerate(GROUPS,1):
        rr=estimates[estimates.review_no.eq(nos[0])]
        if objects!=1 or len(rr)!=1 or any(not (estimates.review_no==no).any() for no in nos):continue
        b=rr[B].to_numpy(float)[0];u=float(rr.uncertainty_px.iloc[0]);query=d.loc[nos,B].to_numpy(float)
        wins=np.zeros(len(nos),int);none=0
        for shift in itertools.product(np.linspace(-u,u,5),repeat=4):
            bb=b+np.array(shift);ov=iou(query,bb)[:,0]
            if max(ov)<.6:none+=1
            else:wins[int(ov.argmax())]+=1
        stability.append(dict(group_id=f'group_{k:02d}',review_nos=','.join(map(str,nos)),scenarios=625,
            no_valid_box_scenarios=none,winner_counts=json.dumps(dict(zip(map(str,nos),map(int,wins)))),
            interpretation='Sensitivity to a uniform coordinate grid; counts are NOT probabilities of hidden labels.'))
    pd.DataFrame(stability).to_csv(A/'audit/duplicate_geometry_stability.csv',index=False)
    s=d.v22_score.to_numpy();t=d.v27_score.to_numpy();reversals=[]
    for i,j in itertools.combinations(range(len(d)),2):
        before=np.sign(s[i]-s[j]);after=np.sign(t[i]-t[j])
        if before!=after:
            a,b=d.iloc[i],d.iloc[j]
            reversals.append(dict(a=int(d.index[i]),b=int(d.index[j]),a_class=a.class_name,b_class=b.class_name,
                a_v22=s[i],b_v22=s[j],a_v27=t[i],b_v27=t[j],a_delta=t[i]-s[i],b_delta=t[j]-s[j],
                a_geometry=a.geometry_status,b_geometry=b.geometry_status,
                a_review=a.visual_status_v29,b_review=b.visual_status_v29))
    rev=pd.DataFrame(reversals);rev.to_csv(A/'audit/v27_rank_reversals.csv',index=False)
    d['v22_rank']=d.v22_score.rank(method='min',ascending=False).astype(int)
    d['v27_rank']=d.v27_score.rank(method='min',ascending=False).astype(int)
    d['rank_change']=d.v27_rank-d.v22_rank
    d['involved_rank_reversals']=[int((rev.a.eq(no)|rev.b.eq(no)).sum()) for no in d.index]
    d.loc[d.geometry_status.eq('robust_error_under_assumptions')&d.v27_delta.lt(-1e-9),'priority_note']='V27_rescue_conflicts_with_visible_geometry_under_stated_uncertainty'
    d.to_csv(A/'audit/all_362_fresh_audit.csv')
    d[d.v27_delta.abs()>1e-9].to_csv(A/'audit/v27_changes_reaudited.csv')
    d[(d.priority_note.ne('')&d.priority_note.notna())|d.group_id.ne('')].to_csv(A/'audit/priority_queries.csv')
    summary=json.loads((A/'audit/summary.json').read_text())
    summary.update(all_measured_queries=int(d.manual_max_iou.notna().sum()),
        all_geometry_counts=d.geometry_status.value_counts().to_dict(),duplicate_groups=len(groups),
        queries_in_duplicate_groups=int(d.group_id.ne('').sum()),v27_rank_pair_changes=len(rev),
        v27_geometry_conflict_queries=d.index[d.geometry_status.eq('robust_error_under_assumptions')&d.v27_delta.lt(-1e-9)].tolist(),
        explanation_of_public_regression='Some public ordering worsened. If every lowered item were truly correct and the only raised item truly erroneous, AP could not decrease. Public membership and exact bad changes remain unknown.')
    (A/'audit/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(pd.DataFrame(stability).to_string(index=False));print('Rank reversals',len(rev))
    print(d[d.v27_delta.abs()>1e-9][['class_name','v22_score','v27_score','rank_change','manual_max_iou','geometry_status']].round(4).to_string())

if __name__=='__main__':run()
