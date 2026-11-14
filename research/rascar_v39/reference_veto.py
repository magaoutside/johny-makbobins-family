"""One fixed cross-representation check; no new thresholds fitted to test.

Only accept a proposed change when the new set reference and the original V31
reference match at the competition's IoU .60 threshold in every retained seed.
Keep the majority's requirement that both fitting days contribute on test.
This deliberately abstains on old/new boundary disagreement; it cannot fix the
old merged-object cases by itself.
"""
from common39 import *
from evaluate import majority_instances
import argparse

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--evidence',default='set_long_identity');args=p.parse_args()
    evidence=pd.read_csv(A/f'features/{args.evidence}_matching_evidence.csv');scores={};audit=[]
    for split,rows in [('oof',q),('test',te)]:
        path=A/(f'results/oof/{args.source}.csv' if split=='oof' else f'results/submissions/submission_{args.source}.csv')
        values=pd.read_csv(path).set_index('id').loc[rows.id].score.to_numpy()
        old={fit:pd.read_csv(ROOT/f'rascar_v31/features/linear_mask_cut0.5_{"oof" if split=="oof" else "test_fit"+fit[-2:]}_evidence.csv').set_index('id') for fit in DAYS}
        for i,r in rows.iterrows():
            if abs(values[i]-BASE[split][i])<1e-10:continue
            ev=majority_instances(evidence[evidence.split.eq(split)&evidence.id.eq(r.id)]);agreements=[]
            for _,v in ev.iterrows():
                bb=old[v.fit_day].loc[r.id,['ref_'+b for b in B]].to_numpy(float);new=v[['ref_'+b for b in B]].to_numpy(float)
                agreements.append(float(iou(bb[None],new[None])[0,0]) if np.isfinite(bb).all() else 0.)
            minimum=min(agreements) if agreements else 0.;veto=minimum<.6
            audit.append(dict(split=split,id=r.id,class_name=r.class_name,min_reference_iou=minimum,veto=veto))
            if veto:values[i]=BASE[split][i]
        scores[split]=values
    name=args.source+'_reference_guard';result=save(name,scores,'Fixed old/new reference agreement IoU >=.60 in all retained seed/day predictions. Abstain on boundary contradictions; no test labels.')
    (A/f'results/{name}.json').write_text(json.dumps(result,indent=2));pd.DataFrame(audit).to_csv(A/f'audit/{name}_vetoes.csv',index=False)

if __name__=='__main__':main()
