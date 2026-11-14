"""Conservative geometry branches, learned references remain cross-day."""
from common18 import *

for weight in [.25,.5]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        split='test' if test else 'oof'
        f=align(A/f'person_body_ridge_{split}.csv',rows)
        seat=align(V17/('seat_features_test_all.csv' if test else 'seat_features_oof.csv'),rows)
        eligible=f.eligible.to_numpy(bool)&(f.height_ratio.to_numpy()<.75)&(f.top_error.to_numpy()<.2)&f.width_ratio.between(.65,1.5).to_numpy()&(f.body_iou.to_numpy()<.6)
        eligible&=seat.back.to_numpy(bool)&(seat.head_count.to_numpy()==1)&(seat.template_n.to_numpy()>=3)&(seat.seat_distance.to_numpy()<1.2)
        eligible&=(rows.y2.to_numpy()<710)&(rows.x1.to_numpy()>5)&(rows.x2.to_numpy()<1275)
        strength=np.clip((.85-f.height_ratio.to_numpy())/.35,0,1)
        # Body and seat templates encode related evidence: replace by max, not sum.
        base16=old.current if test else old.q.score.to_numpy()
        candidate=base16+weight*(1-base16)*strength*eligible
        outputs.append(np.where(eligible,np.maximum(base,candidate),base))
    register(f'body_backbench_w{weight}',*outputs,details=dict(weight=weight,combination='max with existing seat correction',guard='back bench, one head, >=3 templates, seat distance<1.2, no image edge'))

for weight in [.25,.5]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        split='test' if test else 'oof'
        p=align(A/f'wrist_direction_{split}.csv',rows)
        a=align(V17/f'tta_v15_{split}.csv',rows);b=align(V17/f'tta_refined_{split}.csv',rows)
        # Pose must place the query beyond the wrist, with consistent independent views.
        m=rows.class_name.eq('nude_hands').to_numpy()&(p.views.to_numpy()>=3)&(p.distal_fraction.to_numpy()>=.8)&(p.projection.to_numpy()>.2)&(p.projection.to_numpy()<.8)&(p.perp.to_numpy()<.25)
        for f in [a,b]:
            m&=boolean(f.assigned)&(f.overlap.to_numpy()>=.6)&(f.views.to_numpy()>=3)&(f.turns.to_numpy()>=2)&(f.instances.to_numpy()<=1)
        m&=~duplicate_veto(rows,test)
        quality=np.minimum(a.overlap.to_numpy(),b.overlap.to_numpy())
        target=np.clip((.85-quality)/.5,0,1)
        outputs.append(base-weight*np.maximum(base-target,0)*m)
        if weight==.5:pd.DataFrame(dict(id=rows.id,eligible=m,quality=quality,target=target)).to_csv(A/f'pose_hand_guard_{split}.csv',index=False)
    register(f'pose_distal_hand_w{weight}',*outputs,details=dict(weight=weight,minimum_pose_views=3,minimum_distal_fraction=.8,projection_range=[.2,.8],yolo_agreement_iou=.6,duplicate_veto=True))
