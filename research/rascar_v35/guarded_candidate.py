"""Retain established class/duplicate vetoes for temporal rescues.

This is a general evidence gate, not a list of test IDs. A merged pair of cups
cannot be rescued merely because the same merged mask persists over time.
"""
from temporal_context import *
from evaluate import save
import shutil

def permitted(rows,split):
    allowed=np.ones(len(rows),bool)
    sem=pd.read_csv(ROOT/f'rascar_v16/semantic_both_{split}.csv').set_index('id').loc[rows.id]
    st=pd.read_csv(ROOT/f'rascar_v21/object_structure_{split}.csv').set_index('id').loc[rows.id]
    yes=lambda s:s.astype(str).str.lower().isin(['true','1','1.0']).to_numpy()
    for cls in ['cup','tray']:
        ids=np.flatnonzero(rows.class_name.eq(cls).to_numpy())
        allowed[ids]=(sem.iloc[ids]['p_'+cls].to_numpy()>=.8)&st.iloc[ids].distinct_instances.le(1).to_numpy()&~yes(st.iloc[ids].ambiguous_duplicate)
    return allowed

def main():
    out=[];blocked=[]
    for weight in [.125,.25]:
        source=f'temporal_two_nonhands_both_w{weight}';scores={}
        for split,rows in ROWS.items():
            path=A/f'results/oof/{source}.csv' if split=='oof' else A/f'results/submissions/submission_{source}.csv'
            s=pd.read_csv(path).set_index('id').loc[rows.id,'score'].to_numpy()
            veto=(s<BASE[split]-1e-10)&~permitted(rows,split)
            for idx in np.flatnonzero(veto):
                blocked.append(dict(id=rows.iloc[idx].id,split=split,class_name=rows.iloc[idx].class_name,weight=weight,before=float(s[idx]),restored_v31=float(BASE[split][idx])))
            s[veto]=BASE[split][veto];scores[split]=s
        name=f'temporal_two_guarded_w{weight}'
        out.append(save(name,scores,'Two-neighbor temporal geometry; retain pre-existing class/duplicate vetoes for cup/tray score decreases. No test-specific rules.'))
    pd.DataFrame(blocked).to_csv(A/'audit/blocked_rescues.csv',index=False)
    ex=pd.read_csv(A/'results/experiments.csv');ex=ex[~ex.experiment.str.startswith('temporal_two_guarded')]
    ex=pd.concat([ex,pd.DataFrame(out)],ignore_index=True)
    ex.sort_values('pooled_ap',ascending=False).to_csv(A/'results/experiments.csv',index=False)
    # Prefer the smaller modification when both tested weights yield equal AP.
    name='temporal_two_guarded_w0.125'
    src=A/f'results/submissions/submission_{name}.csv';dest=A/'submission_v35_temporal_guarded.csv'
    shutil.copyfile(src,dest)
    m=ex[ex.experiment.eq(name)].iloc[0]
    metadata=dict(file=dest.name,experiment=name,public_ap=None,submitted=False,best_confirmed_public_ap=.96239,
                  baseline=str(PATHS['test'].relative_to(ROOT)),status='research_candidate_not_LB_confirmed',
                  oof_ap=float(m.pooled_ap),day_25=float(m.day_25),day_26=float(m.day_26),changed_test=int(m.changed_test),
                  sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),test_labels_used=False,hands_unchanged=True)
    dest.with_suffix('.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
