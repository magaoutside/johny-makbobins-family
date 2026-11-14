"""A narrow geometry-only variant; no absence penalties on blurred hands.

Undo half of a NEW V32 penalty only if the frozen V22 score <= .2 and two
same-fit-day side heads confirm the box. The threshold is an exploratory,
post-review safeguard, not a calibrated 80% correctness probability.
"""
from confirm_geometry import *

def geometry_agreement(rows,fit):
    refs=[];yes=np.ones(len(rows),bool)
    for kind in ['pool_regional_balanced','pool_regional_balanced_rare']:
        d=align(A/f'features/side_{kind}_fit{fit[-2:]}.csv',rows);yes&=d.valid.to_numpy()&d.presence.ge(.9).to_numpy();refs.append(d[['ref_'+k for k in B]].to_numpy(float))
    yes&=np.array([iou(a,b)[0,0]>=.65 for a,b in zip(*refs)])
    return yes,refs

def confirmed(rows,split,fit):
    aa=align(A/f'features/side_pool_regional_balanced_fit{fit[-2:]}.csv',rows)
    bb=align(A/f'features/side_pool_regional_balanced_rare_fit{fit[-2:]}.csv',rows)
    rb=['ref_'+k for k in B];mutual=np.array([iou(a,b)[0,0] for a,b in zip(aa[rb].to_numpy(float),bb[rb].to_numpy(float))])
    yes=aa.valid.to_numpy()&bb.valid.to_numpy()&aa.presence.ge(.9).to_numpy()&bb.presence.ge(.9).to_numpy()&aa.ref_iou.ge(.65).to_numpy()&bb.ref_iou.ge(.65).to_numpy()&(mutual>=.65)
    for cls in CL:
        ids=np.flatnonzero(rows.class_name.eq(cls).to_numpy());yes[ids]&=guard(split,rows.iloc[ids],cls)
    return yes

def main():
    source='v32_pool_regional_balanced_boundary_conservative_w0.25';outputs=[];ledger=[]
    for split,rows,floor,parentpath in [('oof',q,base_oof,ROOT/'rascar_v32/results/oof/compose_watershed1_confirmed_hand_protection2.csv'),('test',te,current,ROOT/'rascar_v32/submission_v32_research_oof098800.csv')]:
        parent=align(parentpath,rows,'score');path=A/f'results/{"oof" if split=="oof" else "submissions"}/{"" if split=="oof" else "submission_"}{source}.csv';s=align(path,rows,'score')
        if split=='oof':
            gate=np.zeros(len(rows),bool);pg=np.zeros(len(rows),bool)
            for fit in DAYS:
                ids=np.flatnonzero(rows.day.ne(fit).to_numpy());gate[ids]=confirmed(rows.iloc[ids],split,fit);pg[ids],_=geometry_agreement(rows.iloc[ids],fit)
        else:
            gate=confirmed(rows,split,DAYS[0])&confirmed(rows,split,DAYS[1]);pg=np.ones(len(rows),bool);refs=[]
            for fit in DAYS:
                g,rr=geometry_agreement(rows,fit);pg&=g;refs+=rr
            for i in range(len(refs)):
                for j in range(i):pg&=np.array([iou(a,b)[0,0]>=.65 for a,b in zip(refs[i],refs[j])])
        if '--confirmation-only' in sys.argv:s=parent.copy()
        elif '--ungated' not in sys.argv:s=parent+(s-parent)*pg
        gate&=(floor<=.2)&(parent>floor+1e-9)
        delta=.5*(parent-floor)*gate;s=np.clip(s-delta,0,1);outputs.append(s)
        for i in np.flatnonzero(gate):ledger.append(dict(split=split,id=rows.iloc[i].id,class_name=rows.iloc[i].class_name,frozen_v22=float(floor[i]),old_v32=float(parent[i]),new=float(s[i]),confirmation_delta=float(-delta[i])))
    name='v33_conservative_geometry_and_low_score_protection' if '--ungated' in sys.argv else 'v33_consistent_geometry_and_low_score_protection';ss,tt=outputs
    if '--confirmation-only' in sys.argv:name='v33_positive_confirmation_only'
    q.assign(score=ss).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=tt).to_csv(A/f'results/submissions/submission_{name}.csv',index=False)
    policy='Only undo half of V32-only penalties when V22 score<=.2 and both side heads confirm the same box; no new penalties.' if '--confirmation-only' in sys.argv else 'No absence signal; upper-bound IoU penalty requires consistent reference boxes and presence >=.9; low-V22-score positive protection. Same-fit heads on OOF, all four heads on test.'
    out=dict(experiment=name,**metric(q,ss),policy=policy,test_labels_used=False)
    (A/f'results/{name}.json').write_text(json.dumps(out,indent=2));pd.DataFrame(ledger).to_csv(A/f'audit/{name}_restorations.csv',index=False)
    print(json.dumps(out,indent=2));print(pd.DataFrame(ledger).groupby(['split','class_name']).size())
if __name__=='__main__':main()
