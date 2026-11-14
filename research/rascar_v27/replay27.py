"""Exact release-score reproduction from frozen component outputs, offline.

This is score reproduction. Full model inference additionally needs original
images, the pretrained checkpoints and the saved source programs.
"""
from pathlib import Path
import pandas as pd,numpy as np
A=Path(__file__).resolve().parent
for split in ['oof','test']:
    d=pd.read_csv(A/f'replay_inputs/{split}.csv')
    control=(d.assembled+d.merged_delta).clip(0,1).to_numpy()
    candidate=np.clip(control+d.weak_hand_delta,0,1)
    for name,score in [('control',control),('candidate',candidate)]:
        path=A/(f'results/oof/release_{name}.csv' if split=='oof' else f'submission_v27_{name}.csv')
        expected=pd.read_csv(path).set_index('id').loc[d.id,'score'].to_numpy()
        error=float(abs(expected-score).max());assert error<1e-12,(split,name,error)
        print(split,name,'max_error',error)
        if split=='test':pd.DataFrame(dict(id=d.id,score=score)).to_csv(A/f'replayed_submission_v27_{name}.csv',index=False)
