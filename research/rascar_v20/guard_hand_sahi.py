from common20 import *
from evaluate_sahi import old_veto
for variant in ['clean','expanded','frozen']:
    if not (A/f'hand_{variant}_sahi_oof.csv').exists():continue
    fo=align(A/f'hand_{variant}_sahi_oof.csv',q);ft=align(A/f'hand_{variant}_sahi_test.csv',te)
    for weight in [.25,.5]:
        outs=[]
        for rows,base,f,test in [(q,q.score.to_numpy(),fo,False),(te,current,ft,True)]:
            sem=align(ROOT/f'rascar_v16/semantic_tight_{"test" if test else "oof"}.csv',rows,'p_nude_hands')
            gate=rows.class_name.eq('nude_hands').to_numpy()&boolean(f.assigned)&f.instances.eq(1).to_numpy()&f.overlap.ge(.6).to_numpy()&f.confidence.ge(.7).to_numpy()&(sem>=.75)&~old_veto(rows,test)&~boolean(f.duplicate_loser)
            if test:gate &= f.folds.ge(2).to_numpy()
            target=np.clip((.85-f.overlap.to_numpy())/.5,0,1);score=base.copy();score[gate]-=weight*np.maximum(score[gate]-target[gate],0);outs.append(score)
        register(f'hand_{variant}_sahi_guarded_w{weight}',*outs,dict(weight=weight,reference_confidence_min=.7,overlap_min=.6,semantic_min=.75,test_both_fold_support=True,one_to_one_assignment=True))
