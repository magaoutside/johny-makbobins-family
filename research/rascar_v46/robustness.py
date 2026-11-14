"""Development-set robustness, never an independent validation claim."""
from common46 import *
from sklearn.metrics import average_precision_score as AP
import argparse
def main():
    global A
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');p.add_argument('--root',default='rascar_v46');args=p.parse_args();A=ROOT/args.root;q,te,h=inputs();bo,bt=baseline(q,te)
    frames=q.frame_id.unique();rng=np.random.default_rng(4601);groups=[np.flatnonzero(q.frame_id.eq(f)) for f in frames]
    samples=[np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]) for _ in range(2000)]
    base_boot=np.array([AP(q.error.iloc[ix],bo[ix]) for ix in samples]);summary=[]
    for name in args.names:
        s=pd.read_csv(A/f'results/oof/{name}.csv').set_index('id').loc[q.id,'score'].to_numpy();t=pd.read_csv(A/f'results/submissions/{name}.csv').set_index('id').loc[te.id,'score'].to_numpy()
        ds=np.array([AP(q.error.iloc[ix],s[ix]) for ix in samples])-base_boot
        leave=[]
        for fid in frames:
            m=q.frame_id.ne(fid);leave.append(dict(excluded=fid,delta=float(AP(q.loc[m,'error'],s[m])-AP(q.loc[m,'error'],bo[m]))))
        pd.DataFrame(leave).to_csv(A/f'audit/{name}_leave_frame.csv',index=False)
        pairs=[]
        for i in np.flatnonzero(q.error==1):
            for j in np.flatnonzero(q.error==0):
                before=np.sign(bo[i]-bo[j]);after=np.sign(s[i]-s[j])
                if before!=after:pairs.append(dict(error_id=q.id.iloc[i],correct_id=q.id.iloc[j],error_frame=q.frame_id.iloc[i],correct_frame=q.frame_id.iloc[j],before=int(before),after=int(after),status='repaired' if after>before else 'broken'))
        pd.DataFrame(pairs).to_csv(A/f'audit/{name}_pairs.csv',index=False)
        row=dict(name=name,**metrics(q,s),delta=AP(q.error,s)-AP(q.error,bo),frame_bootstrap_low=float(np.quantile(ds,.025)),frame_bootstrap_high=float(np.quantile(ds,.975)),leave_frame_min=min(x['delta'] for x in leave),repaired_pairs=sum(x['status']=='repaired' for x in pairs),broken_pairs=sum(x['status']=='broken' for x in pairs),changed_test=int((abs(t-bt)>1e-9).sum()),note='Repeated development OOF; bootstrap conditional on selection, no independent holdout claim.')
        summary.append(row);(A/f'audit/{name}_stability.json').write_text(json.dumps(row,indent=2));q.assign(old_score=bo,new_score=s,change=s-bo).to_csv(A/f'audit/{name}_all_oof.csv',index=False);te.assign(old_score=bt,new_score=t,change=t-bt).to_csv(A/f'audit/{name}_all_test.csv',index=False)
    pd.DataFrame(summary).to_csv(A/'results/robustness.csv',index=False);print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
