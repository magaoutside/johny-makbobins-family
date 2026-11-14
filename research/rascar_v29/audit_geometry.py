"""Nonofficial visual geometry; never supplies labels to model training."""
from pathlib import Path
import itertools,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v29'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import iou,B,align,te,current,threshold

def interval_iou(query,reference,u):
    # Extrema of rectangle IoU over an axis-aligned coordinate uncertainty box
    # occur at interval ends or at changes in min/max branches (query edges).
    choices=[]
    for j in range(4):
        lo,hi=reference[j]-u,reference[j]+u
        choices.append(sorted(set([lo,hi]+[float(v) for v in query[[j%2,j%2+2]] if lo<=v<=hi])))
    boxes=np.array(list(itertools.product(*choices)))
    boxes=boxes[(boxes[:,2:]>boxes[:,:2]).all(1)]
    values=iou(query,boxes)[0]
    return float(values.min()),float(values.max())

def run():
    m=pd.read_csv(A/'inputs/review_manifest.csv').set_index('review_no')
    v27=align(ROOT/'rascar_v27/submission_v27_candidate.csv',te,'score')
    old=pd.read_csv(ROOT/'rascar_v22/results/reviews/final_reviewed.csv').set_index('id')
    fresh=pd.DataFrame(json.loads((A/'annotations/fresh_review.json').read_text()).values()).set_index('review_no')
    assert set(fresh.index)==set(range(1,363))
    boxes=pd.read_csv(A/'annotations/cup_tight_boxes.csv');rows=[]
    for r in boxes.itertuples():
        query=m.loc[r.review_no,B].to_numpy(float);ref=np.array([getattr(r,c) for c in B])
        lo,hi=interval_iou(query,ref,r.uncertainty_px)
        rows.append(dict(review_no=r.review_no,manual_iou=float(iou(query,ref)[0,0]),iou_min=lo,iou_max=hi,
            uncertainty_px=r.uncertainty_px,reference_note=r.note,**{f'visual_{c}':v for c,v in zip(B,ref)}))
    geom=pd.DataFrame(rows);geom.to_csv(A/'audit/cup_geometry_per_object.csv',index=False)
    summaries=[]
    for no,g in geom.groupby('review_no'):
        low=g.iou_min.max();high=g.iou_max.max();point=g.manual_iou.max()
        state='robust_error_under_assumptions' if high<.6 else 'robust_geometry_under_assumptions' if low>=.6 else 'boundary_uncertain'
        summaries.append(dict(review_no=no,manual_max_iou=point,iou_min=low,iou_max=high,geometry_status=state,
            visual_reference_count=len(g),official_test_label=False))
    gs=pd.DataFrame(summaries).set_index('review_no')
    out=m.join(fresh).join(gs.drop(columns='official_test_label'))
    out['v22_score']=pd.Series(current,index=te.id).reindex(out.id).to_numpy()
    out['v27_score']=pd.Series(v27,index=te.id).reindex(out.id).to_numpy()
    out['v27_delta']=out.v27_score-out.v22_score
    out['old_visual_status']=old.visual_status.reindex(out.id).to_numpy()
    out['diagnostic_threshold']=threshold
    out['priority_note']=''
    clear=out.visual_status_v29.eq('clear_error')|out.geometry_status.eq('robust_error_under_assumptions')
    out.loc[clear&out.v22_score.lt(threshold),'priority_note']='clear_or_geometrically_robust_error_below_diagnostic_threshold'
    out.loc[out.visual_status_v29.eq('plausible_correct')&out.v22_score.ge(threshold),'priority_note']='plausible_correct_high_score_requires_measurement'
    out.loc[out.v27_delta.lt(-1e-9)&out.manual_max_iou.lt(.6),'priority_note']='V27_lowered_score_but_tight_box_estimate_is_below_0.6'
    out.to_csv(A/'audit/all_362_fresh_audit.csv')
    out[out.priority_note.ne('')].to_csv(A/'audit/priority_queries.csv')
    changed=out[abs(out.v27_delta)>1e-9]
    changed.to_csv(A/'audit/v27_changes_reaudited.csv')
    stats=dict(reviewed=362,frames=m.frame_id.nunique(),class_counts=m.class_name.value_counts().to_dict(),
        statuses=out.visual_status_v29.value_counts().to_dict(),issues=out.issue_v29.value_counts().to_dict(),
        geometry_counts=gs.geometry_status.value_counts().to_dict(),measured_cup_queries=len(gs),
        v27_lowered_cups=int((changed.class_name.eq('cup')&changed.v27_delta.lt(0)).sum()),
        v27_lowered_cups_point_iou_below_06=int((changed.class_name.eq('cup')&changed.v27_delta.lt(0)&changed.manual_max_iou.lt(.6)).sum()),
        v27_lowered_cups_robust_error=int((changed.class_name.eq('cup')&changed.v27_delta.lt(0)&changed.geometry_status.eq('robust_error_under_assumptions')).sum()),
        clear_low_diagnostic_count=int((clear&out.v22_score.lt(threshold)).sum()),
        limitations=['Single-model visual review, not hidden test ground truth.',
        'IoU intervals assume visible/amodal cup bounds estimated here and the stated coordinate uncertainty; they are not statistical confidence intervals.',
        'Correct geometric overlap does not imply correct one-to-one assignment.',
        'No visual audit labels used to train or calibrate models.'])
    (A/'audit/summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    print(changed[changed.class_name.eq('cup')][['v22_score','v27_score','manual_max_iou','iou_min','iou_max','geometry_status']].round(4).to_string())
    print('PRIORITY',out.loc[clear&out.v22_score.lt(threshold),['class_name','v22_score','v27_score','issue_v29','manual_max_iou']].to_string())

if __name__=='__main__':run()
