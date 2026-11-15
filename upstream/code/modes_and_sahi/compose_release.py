"""Freeze V20 candidates; keep V18 and rejected ablations intact."""
from common20 import *
from evaluate_sahi import old_veto

fo=pd.concat([pd.read_csv(A/f'person_dino_mode_held_{fit}.csv') for fit in DAYS]).set_index('id').loc[q.id].reset_index()
tt=[align(A/f'person_dino_mode_test_{fit}.csv',te) for fit in DAYS];ft=(tt[0]+tt[1])/2
for name,hand_weight in [('release_main',.5),('release_conservative',.25)]:
    outputs=[]
    for rows,base,f,test in [(q,q.score.to_numpy(),fo,False),(te,current,ft,True)]:
        split='test' if test else 'oof';front_p=align(A/f'person_dino_modes_logreg_{split}.csv',rows,'probability');back_p=align(A/f'person_dino_modes_catboost_{split}.csv',rows,'probability')
        person=rows.class_name.eq('person').to_numpy();eligible=person&f.eligible.eq(1).to_numpy()&f.support.ge(3).to_numpy()&f.edge.eq(0).to_numpy()
        front=f.mode_front.ge(.6).to_numpy()
        if test:front &= tt[0].mode_front.ge(.5).to_numpy()&tt[1].mode_front.ge(.5).to_numpy()
        rescue=eligible&front&f.visible_outside.lt(.01).to_numpy()&(base>.6)&(front_p<base)&f.position_distance.lt(1.5).to_numpy()&f.width_ratio.between(.65,1.5).to_numpy()&f.iou_q75.ge(.55).to_numpy()
        short=eligible&f.mode_back.ge(.75).to_numpy()&f.bottom_gap.gt(.18).to_numpy()&f.height_ratio.lt(.8).to_numpy()&f.top_gap.abs().lt(.1).to_numpy()&(back_p>base)
        score=base.copy();score[rescue]=.7*score[rescue]+.3*front_p[rescue];score[short]=.85*score[short]+.15*back_p[short]
        hands=align(A/f'hand_expanded_sahi_{split}.csv',rows);sem=align(ROOT/f'geometry_guards/semantic_tight_{split}.csv',rows,'p_nude_hands')
        gate=rows.class_name.eq('nude_hands').to_numpy()&boolean(hands.assigned)&hands.instances.eq(1).to_numpy()&hands.overlap.ge(.6).to_numpy()&hands.confidence.ge(.7).to_numpy()&(sem>=.75)&~old_veto(rows,test)&~boolean(hands.duplicate_loser)
        if test:gate &= hands.folds.ge(2).to_numpy()
        target=np.clip((.85-hands.overlap.to_numpy())/.5,0,1);score[gate]-=hand_weight*np.maximum(score[gate]-target[gate],0)
        outputs.append(score)
        pd.DataFrame(dict(id=rows.id.to_numpy(),baseline=base,front_mode_rescue=rescue,back_short_signal=short,sahi_hand_rescue=gate,front_mode=f.mode_front.to_numpy(),back_mode=f.mode_back.to_numpy(),standing_mode=f.mode_standing.to_numpy(),sahi_iou=hands.overlap.to_numpy(),sahi_confidence=hands.confidence.to_numpy(),score=score)).to_csv(A/f'{name}_evidence_{split}.csv',index=False)
    register(name,*outputs,dict(baseline='V18 LB 0.95259',front_person='DINO mode + logistic geometry expert, weight .3',back_person='DINO mode + CatBoost geometry expert, weight .15',standing='abstain from new seat correction; V18 ranking retained',hands='expanded YOLO26; independent 256/384 SAHI; confidence >=.7; unique assignment',hand_weight=hand_weight,cup_tray='V18 retained: new experiments failed audit'))
    (A/f'submission_v20_{"main" if name=="release_main" else "conservative"}.csv').write_bytes((R/'submissions'/f'submission_{name}.csv').read_bytes())
