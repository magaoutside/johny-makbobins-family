"""Package measured results, audit evidence, and replay inputs."""
from common20 import *
from evaluate_sahi import old_veto
import sys,platform,importlib.metadata,html,zipfile

(A/'replay_inputs').mkdir(exist_ok=True)
fo=pd.concat([pd.read_csv(A/f'person_dino_mode_held_{fit}.csv') for fit in DAYS]).set_index('id').loc[q.id].reset_index()
tt=[align(A/f'person_dino_mode_test_{fit}.csv',te) for fit in DAYS];ft=(tt[0]+tt[1])/2
for split,rows,base,f,test in [('oof',q,q.score.to_numpy(),fo,False),('test',te,current,ft,True)]:
    d=f.reset_index(drop=True).copy();d['id']=rows.id.to_numpy();d['class_name']=rows.class_name.to_numpy();d['baseline']=base
    d['front_probability']=align(A/f'person_dino_modes_logreg_{split}.csv',rows,'probability')
    d['back_probability']=align(A/f'person_dino_modes_catboost_{split}.csv',rows,'probability')
    d['both_front_agree']=(tt[0].mode_front.ge(.5).to_numpy()&tt[1].mode_front.ge(.5).to_numpy()) if test else True
    hand=align(A/f'hand_expanded_sahi_{split}.csv',rows)
    for col in ['assigned','instances','overlap','confidence','duplicate_loser']:d['hand_'+col]=hand[col].to_numpy()
    d['hand_fold_agreement']=hand.folds.ge(2).to_numpy() if test else True
    d['semantic_hand']=align(ROOT/f'geometry_guards/semantic_tight_{split}.csv',rows,'p_nude_hands')
    d['old_duplicate_veto']=old_veto(rows,test)
    if not test:d['day']=rows.day.to_numpy();d['error']=rows.error.to_numpy()
    d.to_csv(A/f'replay_inputs/{split}.csv',index=False)
