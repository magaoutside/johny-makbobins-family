"""Conservative application of uncertain mode experts; no audit labels fitted."""
from common20 import *
fo=pd.concat([pd.read_csv(A/f'person_mode_held_{fit}.csv') for fit in DAYS]).set_index('id').loc[q.id].reset_index()
tt=[align(A/f'person_mode_test_{fit}.csv',te) for fit in DAYS]
ft=(tt[0]+tt[1])/2
for family in ['logreg','catboost']:
    po=align(A/f'person_modes_{family}_oof.csv',q,'probability');pt=align(A/f'person_modes_{family}_test.csv',te,'probability')
    for weight in [.15,.3]:
        outs=[]
        for rows,base,pred,fe,test in [(q,q.score.to_numpy(),po,fo,False),(te,current,pt,ft,True)]:
            eligible=rows.class_name.eq('person').to_numpy()&fe.eligible.eq(1).to_numpy()&fe.support.ge(3).to_numpy()&fe.edge.eq(0).to_numpy()
            # Rescue only clearly front-seated people, with no visible limb
            # disagreement. Require both day experts to agree on test modes.
            front=fe.mode_front.ge(.65).to_numpy()
            if test:front &= tt[0].mode_front.ge(.55).to_numpy()&tt[1].mode_front.ge(.55).to_numpy()
            rescue=eligible&front&fe.visible_outside.lt(.01).to_numpy()&(base>.6)&(pred<base)&fe.position_distance.lt(1.5).to_numpy()
            # Do not raise errors merely because a body is longer than a seat
            # template. Explicit lower-bound mismatch is required.
            penalize=eligible&fe.mode_back.ge(.7).to_numpy()&fe.mode_confidence.ge(.75).to_numpy()&fe.bottom_gap.gt(.18).to_numpy()&fe.height_ratio.lt(.8).to_numpy()&fe.top_gap.abs().lt(.1).to_numpy()&(pred>base)
            score=base.copy();m=rescue|penalize;score[m]=(1-weight)*score[m]+weight*pred[m];outs.append(score)
        register(f'person_modes_guarded_{family}_w{weight}',*outs,dict(weight=weight,guard='two fold front-seated consensus for rescue; bottom undercoverage for increasing error; no audit labels in fitting'))
