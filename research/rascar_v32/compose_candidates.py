"""Composition ablations; no fitting, ID-specific rules, or leaderboard use.

These inherit V31's model-selection bias. DeepLab only confirms a good box;
missing/poor segmentation never increases an error score in this composition.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v32';V31=ROOT/'rascar_v31'
sys.path.insert(0,str(A))
from consensus import save,q,te,base_oof,current,align
from common25 import guard
import numpy as np,pandas as pd

def main():
    out=[];ledger=[]
    wide='linear_mask_cut0.5_all_two_sided_w0.25';hands='watershed_fused_cut0.35_nude_hands_penalty_only_w0.25'
    for use_watershed in [False,True]:
        for protect_confirmed in [0,1,2]:
            scores=[]
            for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
                p=V31/f'results/{"oof" if split=="oof" else "submissions"}/{"" if split=="oof" else "submission_"}{wide}.csv'
                s=align(p,rows).score.to_numpy().copy();mask=rows.class_name.eq('nude_hands').to_numpy()
                if use_watershed:
                    p=A/f'results/{"oof" if split=="oof" else "submissions"}/{"" if split=="oof" else "submission_"}{hands}.csv'
                    s[mask]=align(p,rows).score.to_numpy()[mask]
                if protect_confirmed:
                    e=align(A/f'features/handseg_cut0.5_{split}_evidence.csv',rows.loc[mask])
                    confirm=(e.near_iou.ge(.65)&e.far_iou.ge(.65)&e.mutual_iou.ge(.65)&e.near_confidence.ge(.8)&e.far_confidence.ge(.8)&~e.near_touch&~e.far_touch).to_numpy()
                    # A well-localized hand may still be a duplicate or have a
                    # structural/classification defect. Reuse the existing V22 vetoes.
                    if protect_confirmed==2:confirm&=guard(split,rows.loc[mask],'nude_hands')
                    ids=np.flatnonzero(mask)[confirm];s[ids]=np.minimum(s[ids],base[ids])
                    for i in ids:ledger.append(dict(split=split,id=rows.iloc[i].id,use_watershed=use_watershed,protection_policy=protect_confirmed,base_score=float(base[i]),new_score=float(s[i])))
                scores.append(s)
            name=f'compose_watershed{int(use_watershed)}_confirmed_hand_protection{int(protect_confirmed)}'
            out.append(save(name,*scores))
    pd.DataFrame(out).to_csv(A/'results/composition_experiments.csv',index=False);pd.DataFrame(ledger).to_csv(A/'audit/confirmed_hand_protection.csv',index=False)
    (A/'audit/composition_protocol.json').write_text(json.dumps(dict(hands_source=hands,other_classes_source=wide,variants=6,
        confirmed_hand='Both DeepLab contexts 2.4x/4x have query IoU>=.65, mutual IoU>=.65, mean foreground probability>=.8, no touching crop edge. Only cancel a NEW penalty, never lower below V22.',
        thresholds='Fixed margins around official IoU .6; no ID-specific exceptions.',
        protection_policy_2='Added after inspecting an OOF false rescue: reuse prior V22 semantic/duplicate/structure rescue guard. This is post-hoc algorithm development, not untouched validation.',
        missing_hand_segmentation='Abstain',test_gt_used_for_fit=False,leaderboard_upload=False),indent=2))
    print(pd.DataFrame(out).to_string(index=False))
if __name__=='__main__':main()
