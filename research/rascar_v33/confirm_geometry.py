"""Use agreement to undo only NEW V32 penalties, preserving V22 as a floor.

Two architectures/data ablations from the SAME fitting day must agree on OOF.
Test also requires the second day's pair to agree. No test IDs in the rule.
"""
from evaluate import *

def gates(rows,split,fit):
    a=align(A/f'features/side_regional_hardneg_fit{fit[-2:]}.csv',rows)
    b=align(A/f'features/side_pool_regional_fit{fit[-2:]}.csv',rows)
    rb=['ref_'+k for k in B];mutual=np.array([iou(x,y)[0,0] for x,y in zip(a[rb].to_numpy(float),b[rb].to_numpy(float))])
    allowed=a.valid.to_numpy()&b.valid.to_numpy()&a.presence.ge(.9).to_numpy()&b.presence.ge(.9).to_numpy()&a.ref_iou.ge(.65).to_numpy()&b.ref_iou.ge(.65).to_numpy()&(mutual>=.65)
    for cls in CL:
        ids=np.flatnonzero(rows.class_name.eq(cls).to_numpy());allowed[ids]&=guard(split,rows.iloc[ids],cls)
    return allowed,pd.DataFrame(dict(id=rows.id,fit_day=fit,head1_iou=a.ref_iou.to_numpy(),head2_iou=b.ref_iou.to_numpy(),mutual_iou=mutual,allowed=allowed))

def main():
    s=align(ROOT/'rascar_v32/results/oof/compose_watershed1_confirmed_hand_protection2.csv',q,'score');t=align(ROOT/'rascar_v32/submission_v32_research_oof098800.csv',te,'score')
    og=np.zeros(len(q),bool);tg=np.ones(len(te),bool);evidence=[]
    for fit in DAYS:
        ids=np.flatnonzero(q.day.ne(fit).to_numpy());og[ids],e=gates(q.iloc[ids],'oof',fit);e['split']='oof';evidence.append(e)
        g,e=gates(te,'test',fit);tg&=g;e['split']='test';evidence.append(e)
    pd.concat(evidence).to_csv(A/'audit/positive_geometry_confirmation.csv',index=False)
    out=[];ledger=[];audit=align(ROOT/'rascar_v29/audit/all_362_fresh_audit.csv',te)
    for fraction in [.5,1.]:
        ss=s.copy();tt=t.copy();oi=og&(s>base_oof);ti=tg&(t>current)
        ss[oi]-=fraction*(s[oi]-base_oof[oi]);tt[ti]-=fraction*(t[ti]-current[ti]);name=f'v32_positive_geometry_restore{fraction}'
        q.assign(score=ss).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=tt).to_csv(A/f'results/submissions/submission_{name}.csv',index=False)
        out.append(dict(experiment=name,**metric(q,ss),changed_oof=int(oi.sum()),changed_test=int(ti.sum()),known_error_down=int(((audit.visual_status_v29.isin(['clear_error','likely_error'])|audit.geometry_status.eq('robust_error_under_assumptions')).to_numpy()&(tt<t-.02)).sum()),plausible_correct_down=int((audit.visual_status_v29.eq('plausible_correct').to_numpy()&(tt<t-.02)).sum())))
        for split,rows,old,new,mask in [('oof',q,s,ss,oi),('test',te,t,tt,ti)]:
            for i in np.flatnonzero(mask):ledger.append(dict(experiment=name,split=split,id=rows.iloc[i].id,class_name=rows.iloc[i].class_name,old=float(old[i]),new=float(new[i])))
    pd.DataFrame(out).to_csv(A/'results/confirmation_experiments.csv',index=False);pd.DataFrame(ledger).to_csv(A/'audit/confirmation_changes.csv',index=False)
    print(pd.DataFrame(out).to_string(index=False));z=pd.DataFrame(ledger);z=z[z.split.eq('test')&z.experiment.eq('v32_positive_geometry_restore1.0')].merge(audit.reset_index()[['id','review_no','visual_status_v29']],on='id');print(z.to_string(index=False))
if __name__=='__main__':main()
