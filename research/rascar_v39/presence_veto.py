"""Keep inherited numerical semantic/duplicate safeguards for small objects."""
from common39 import *
from evaluate_correspondence import guard
import argparse

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);args=parser.parse_args();scores={};decisions=[]
    for split,rows in [('oof',q),('test',te)]:
        path=A/(f'results/oof/{args.source}.csv' if split=='oof' else f'results/submissions/submission_{args.source}.csv')
        candidate=pd.read_csv(path).set_index('id').loc[rows.id].score.to_numpy();allowed=guard(rows,split)
        sem=pd.read_csv(ROOT/f'rascar_v16/semantic_both_{split}.csv').set_index('id')
        for i,r in rows.iterrows():
            if r.class_name=='person' or abs(candidate[i]-BASE[split][i])<1e-10:continue
            presence=float(sem.loc[r.id,'p_'+r.class_name]);veto=presence<.8 or (candidate[i]<BASE[split][i] and not allowed[i])
            decisions.append(dict(split=split,id=r.id,class_name=r.class_name,veto=bool(veto),semantic_presence=presence,structure_allowed=bool(allowed[i])))
            if veto:candidate[i]=BASE[split][i]
        scores[split]=candidate
    name=args.source+'_presence_guard';result=save(name,scores,'Inherited V37 semantic presence >=.8 and numerical structure/duplicate protections; same rule on OOF and test, no visual status or test labels.')
    (A/f'results/{name}.json').write_text(json.dumps(result,indent=2));pd.DataFrame(decisions).to_csv(A/f'audit/{name}_vetoes.csv',index=False)

if __name__=='__main__':main()
