"""Standalone V18 scoring from frozen cross-day evidence, no GPU or Kaggle needed."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
A=Path(__file__).resolve().parent

def predict(d,seat_weight=.55,geometry_weight=.25,absence_weight=.5):
    b=d.base.to_numpy(float);b16=d.base16.to_numpy(float)
    person=d.class_name.eq('person').to_numpy();hand=d.class_name.eq('nude_hands').to_numpy()
    body=d.body_eligible.to_numpy(bool)&(d.body_height_ratio<.75)&(d.body_top_error<.2)&d.body_width_ratio.between(.65,1.5)&(d.body_body_iou<.6)
    body&=d.seat_back.astype(bool)&d.seat_head_count.eq(1)&d.seat_template_n.ge(3)&d.seat_seat_distance.lt(1.2)&d.y2.lt(710)&d.x1.gt(5)&d.x2.lt(1275)
    strength=np.clip((.85-d.body_height_ratio)/.35,0,1)
    people=np.where(body,np.maximum(b,b16+.5*(1-b16)*strength),b)
    people=np.maximum(people,b+d.seat_delta04.to_numpy()*(seat_weight/.4-1))
    geom=hand&d.geom_reference_iou.ge(.3)&d.geom_center_instances.eq(1)&d.tight_p_nude_hands.ge(.75)&d.cal_assigned&~d.duplicate_veto
    hands=b-geometry_weight*np.maximum(b-d.geom_prediction.to_numpy(),0)*geom
    distal=hand&d.wrist_views.ge(3)&d.wrist_distal_fraction.ge(.8)&d.wrist_projection.gt(.2)&d.wrist_projection.lt(.8)&d.wrist_perp.lt(.25)&~d.duplicate_veto
    for p in ['v15','refined']:
        distal&=d[p+'_assigned']&d[p+'_overlap'].ge(.6)&d[p+'_views'].ge(3)&d[p+'_turns'].ge(2)&d[p+'_instances'].le(1)
    quality=np.minimum(d.v15_overlap,d.refined_overlap)
    hands=np.minimum(hands,b-.5*np.maximum(b-np.clip((.85-quality)/.5,0,1),0)*distal)
    proximal=hand&d.wrist_views.ge(2)&d.wrist_proximal_fraction.ge(.75)&d.wrist_projection.lt(-.1)&d.wevis_hand_iou.lt(.3)
    assert not (proximal&(hands<b-1e-9)).any()
    score=people+hands-b+.25*(1-b)*proximal
    absence=person&d.wevis_head_count.eq(0)&d.pose_pose_best_iou.lt(.2)&d.pose_pose_confirmed_views.eq(0)&d.tight_p_person.lt(.5)&d.context_p_person.lt(.5)
    merged=hand&d.v15_instances.ge(2)&d.refined_instances.ge(2)&d.refined_overlap.lt(.6)&d.wevis_hand_iou.lt(.6)
    assert not ((absence|merged)&(abs(score-b)>1e-9)).any()
    score+=absence_weight*(1-b)*absence
    before=(b-.25)/.75
    score+=np.where(merged,before+.5*(1-before)-b,0)
    assert np.isfinite(score).all() and np.asarray((score>=0)&(score<=1)).all()
    return np.asarray(score)

if __name__=='__main__':
    from sklearn.metrics import average_precision_score
    out=A/'replayed';out.mkdir(exist_ok=True);checks=[]
    for name,params in [('main',{}),('conservative',dict(seat_weight=.4,geometry_weight=0,absence_weight=.25))]:
        for split in ['oof','test']:
            d=pd.read_csv(A/'replay_inputs'/f'{split}.csv');p=predict(d,**params)
            expected=pd.read_csv(A/'results'/('oof' if split=='oof' else 'submissions')/(('' if split=='oof' else 'submission_')+'release_'+name+'.csv'))
            assert d.id.tolist()==expected.id.tolist()
            err=float(np.max(abs(p-expected.score.to_numpy())));assert err<1e-12
            pd.DataFrame(dict(id=d.id,score=p)).to_csv(out/f'{name}_{split}.csv',index=False)
            rec=dict(variant=name,split=split,rows=len(d),max_abs_difference=err)
            if split=='oof':rec['pooled_ap']=float(average_precision_score(d.error,p))
            checks.append(rec)
    (A/'replay_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
