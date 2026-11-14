from common24 import *

if __name__=='__main__':
    # Appearance can confirm good geometry, but must not erase a geometric error.
    for family in ['border_cb','context_cb','context_trees']:
        for overlap in [.6,.65]:
            outputs=[]
            for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
                p=align(A/f'features/{family}_{split}.csv',rows)
                f=align(A/f'features/instances_{split}.csv',rows)
                gate=rows.class_name.eq('nude_hands').to_numpy()&guard(split,rows,'nude_hands')
                gate&=f.assigned.eq(1).to_numpy()&f.rival_margin.ge(.1).to_numpy()&f.reference_iou.ge(overlap).to_numpy()&f.ref_conf.ge(.55).to_numpy()
                gate&=p.probability.lt(.4).to_numpy()
                if split=='test':gate&=p[['p25','p26']].max(axis=1).lt(.5).to_numpy()
                score=base-.5*np.maximum(base-p.probability.to_numpy(),0)*gate
                outputs.append(score)
            register(f'guarded_hand_{family}_iou{overlap}',*outputs,dict(type='appearance_with_geometry_gate',family=family,iou_min=overlap,weight=.5))
    for blend in ['mean','minimum','agreement']:
        outputs=[]
        for split,rows in [('oof',q),('test',te)]:
            old=align(ROOT/('rascar_v23/results/oof/combined_catboost_cup_rescue_w0.4.csv' if split=='oof' else 'rascar_v23/submission_v23_cv_cup_candidate.csv'),rows,'score')
            new=align(A/(f'results/oof/context_cb_cup_rescue_w0.5.csv' if split=='oof' else 'results/submissions/submission_context_cb_cup_rescue_w0.5.csv'),rows,'score')
            result=(old+new)/2 if blend=='mean' else np.minimum(old,new) if blend=='minimum' else np.maximum(old,new)
            outputs.append(result)
        register(f'cup_experts_{blend}',*outputs,dict(type='cup_cv_expert_ensemble',blend=blend,experts=['V23 context relations CatBoost','V24 boundary HOG CatBoost']))
    # Research-only hand branch keeps a separately inspectable score/case tradeoff.
    outputs=[]
    for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
        cup=align(A/(f'results/oof/cup_experts_mean.csv' if split=='oof' else 'results/submissions/submission_cup_experts_mean.csv'),rows,'score')
        hand=align(A/(f'results/oof/context_trees_nude_hands_rescue_w0.5.csv' if split=='oof' else 'results/submissions/submission_context_trees_nude_hands_rescue_w0.5.csv'),rows,'score')
        outputs.append(np.clip(cup+hand-base,0,1))
    register('cup_mean_hand_research',*outputs,dict(type='research_tradeoff',not_recommended_as_main=True,reason='Hand visual cases improve, cross-day AP drops'))
    outputs=[]
    for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
        cup=align(A/(f'results/oof/cup_experts_minimum.csv' if split=='oof' else 'results/submissions/submission_cup_experts_minimum.csv'),rows,'score')
        hand=align(A/(f'results/oof/guarded_hand_context_trees_iou0.65.csv' if split=='oof' else 'results/submissions/submission_guarded_hand_context_trees_iou0.65.csv'),rows,'score')
        outputs.append(np.clip(cup+hand-base,0,1))
    register('cup_min_guarded_hand_research',*outputs,dict(type='research_tradeoff',not_recommended_as_main=True,reason='Hand gate has zero OOF coverage; one subjective test case improves'))
