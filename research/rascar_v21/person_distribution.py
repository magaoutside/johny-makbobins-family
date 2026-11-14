"""Use the distribution of fit-day human templates, not a single extension."""
from common21 import *
fo=pd.concat([pd.read_csv(ROOT/f'rascar_v20/person_dino_mode_held_{fit}.csv') for fit in DAYS]).set_index('id').loc[q.id].reset_index()
tt=[align(ROOT/f'rascar_v20/person_dino_mode_test_{fit}.csv',te) for fit in DAYS];ft=(tt[0]+tt[1])/2
for weight in [.25,.5]:
    out=[]
    for rows,base,f,test in [(q,q.score.to_numpy(),fo,False),(te,current,ft,True)]:
        m=(rows.class_name.eq('person').to_numpy()&f.eligible.eq(1).to_numpy()&f.support.ge(4).to_numpy()
           &f.mode_back.ge(.8).to_numpy()&f.edge.eq(0).to_numpy()&f.bottom_gap.gt(.15).to_numpy()
           &f.top_gap.abs().lt(.1).to_numpy()&f.height_ratio.lt(.85).to_numpy()&f.valid_fraction.lt(.4).to_numpy()
           &f.iou_q75.lt(.67).to_numpy()&f.width_ratio.between(.8,1.3).to_numpy())
        if test:m &= tt[0].mode_back.ge(.7).to_numpy()&tt[1].mode_back.ge(.7).to_numpy()
        target=1-f.valid_fraction.to_numpy();s=base.copy();s[m]+=weight*np.maximum(target[m]-s[m],0);out.append(s)
        pd.DataFrame(dict(id=rows.id.to_numpy(),gate=m,target=target,new_score=s)).to_csv(A/f'person_distribution_w{weight}_{"test" if test else "oof"}.csv',index=False)
    register(f'person_distribution_w{weight}',*out,dict(weight=weight,mode='back seated',target='one minus weighted fraction of fitting-day templates with IoU >=.6',guard='lower boundary gap, same upper boundary, modest width difference, two-fold mode agreement'))
