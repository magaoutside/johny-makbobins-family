"""Only accept robust small-object geometry from the final refitted heads.
Presence/duplicate vetoes use frozen numerical model evidence, no test labels.
"""
from common37 import *
from evaluate_correspondence import guard

def main():
    pred=pd.read_csv(A/'features/refit_predictions.csv');results=[];audit=[]
    for mode in ['daily_consensus','full_fit']:
        for cls in ['nude_hands','cup','tray']:
            scores={}
            for split,rows in ROWS.items():
                name=f'refit_{mode}_median_all';path=A/(f'results/oof/{name}.csv' if split=='oof' else f'results/submissions/submission_{name}.csv')
                raw=pd.read_csv(path).set_index('id').loc[rows.id].score.to_numpy();s=BASE[split].copy();g=guard(rows,split)
                sem=pd.read_csv(ROOT/f'rascar_v16/semantic_both_{split}.csv').set_index('id')
                for i,r in rows.iterrows():
                    if r.class_name!=cls or abs(raw[i]-s[i])<1e-10:continue
                    pp=pred[pred.id.eq(r.id)&pred.split.eq(split)]
                    if split=='test':pp=pp[pp.fit_day.eq('all') if mode=='full_fit' else pp.fit_day.ne('all')]
                    refs=pp[['ref_'+b for b in B]].to_numpy();valid=np.isfinite(refs).all() and len(refs)>0
                    mutual=float(iou(refs,refs).min()) if valid else 0.
                    semantic=float(sem.loc[r.id,'p_'+cls])
                    valid=valid and mutual>=.65
                    if raw[i]<s[i]:valid=valid and pp.overlap.min()>=.65 and g[i] and semantic>=.8
                    else:valid=valid and pp.overlap.max()<.5 and semantic>=.8
                    audit.append(dict(mode=mode,split=split,id=r.id,class_name=cls,allowed=bool(valid),reference_agreement=mutual,semantic=semantic,v31_score=s[i],raw_score=raw[i]))
                    if valid:s[i]=raw[i]
                scores[split]=s
            results.append(save(f'object_refit_{mode}_{cls}',scores,'Three-seed agreement on reference >=.65; positive presence and duplicate veto; rescue IoU>=.65, penalty IoU<.5.'))
    pd.DataFrame(results).to_csv(A/'results/object_refit_experiments.csv',index=False);pd.DataFrame(audit).to_csv(A/'audit/object_refit_gates.csv',index=False)

if __name__=='__main__':main()
