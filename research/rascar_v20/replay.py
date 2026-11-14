"""Rebuild V20 ranking from frozen cross-day evidence, without GPU or old folders.

These inputs are cached model predictions, not retraining data. Full training
and inference sources are preserved alongside this script.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
A=Path(__file__).resolve().parent
assert hashlib.sha256((A/'baseline_lb095259.csv').read_bytes()).hexdigest()=='27d4ce779b8d8d26e6df01c9eccef83f2a54ca99e19cbbe0099b2ca4271dd2bd'

def rebuild(d,hand_weight):
    base=d.baseline.to_numpy();score=base.copy()
    person=d.class_name.eq('person')
    eligible=person&d.eligible.eq(1)&d.support.ge(3)&d.edge.eq(0)
    front=(eligible&d.mode_front.ge(.6)&d.both_front_agree&d.visible_outside.lt(.01)
           &d.baseline.gt(.6)&(d.front_probability<d.baseline)&d.position_distance.lt(1.5)
           &d.width_ratio.between(.65,1.5)&d.iou_q75.ge(.55))
    short=(eligible&d.mode_back.ge(.75)&d.bottom_gap.gt(.18)&d.height_ratio.lt(.8)
           &d.top_gap.abs().lt(.1)&(d.back_probability>d.baseline))
    score[front]=.7*score[front]+.3*d.loc[front,'front_probability']
    score[short]=.85*score[short]+.15*d.loc[short,'back_probability']
    hand=(d.class_name.eq('nude_hands')&d.hand_assigned&d.hand_instances.eq(1)
          &d.hand_overlap.ge(.6)&d.hand_confidence.ge(.7)&d.semantic_hand.ge(.75)
          &~d.old_duplicate_veto&~d.hand_duplicate_loser&d.hand_fold_agreement)
    target=np.clip((.85-d.hand_overlap.to_numpy())/.5,0,1)
    score[hand]-=hand_weight*np.maximum(score[hand]-target[hand],0)
    return score

def main():
    output=A/'replayed';output.mkdir(exist_ok=True)
    for name,weight in [('main',.5),('conservative',.25)]:
        for split,n in [('oof',198),('test',362)]:
            d=pd.read_csv(A/f'replay_inputs/{split}.csv',float_precision='round_trip')
            s=rebuild(d,weight)
            assert len(d)==n and d.id.is_unique and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
            expected=pd.read_csv(A/f'results/{"submissions/submission_" if split=="test" else "oof/"}release_{name}.csv').set_index('id').loc[d.id,'score'].to_numpy()
            assert np.max(np.abs(s-expected))<1e-12
            d[['id']].assign(score=s).to_csv(output/f'{split}_{name}.csv',index=False)
            if split=='oof':
                print(name,'OOF',average_precision_score(d.error,s))
                for day,g in d.groupby('day'):print(day,average_precision_score(g.error,s[g.index]))
    print('Verified: 362 unique test ids; finite scores in [0,1]; both releases reproduced within 1e-12.')

if __name__=='__main__':main()
