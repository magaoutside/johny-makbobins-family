"""Replay the frozen V57 score correction; no training or test annotations."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
A=Path(__file__).resolve().parent
d=np.load(A/'replay_inputs.npz',allow_pickle=False)
b=d['baseline'];c=d['class_name']
s=b+np.clip(.2*(d['p_full']-b),0,.1)+np.clip(.2*(d['p_crop']-b),-.1,0)
m=c=='nude_hands';s[m]=b[m]+np.clip(.2*(d['p_anchor_adapted'][m]-b[m]),-.1,.1)
m=c=='tray';x=np.clip(b[m],.0001,.9999);s[m]=1/(1+np.exp(-(np.log(x/(1-x))-.4)))
s=np.clip(s,0,1)
assert np.array_equal(s,d['expected']),float(abs(s-d['expected']).max())
m=d['split']=='test';out=A/'replay_submission.csv'
pd.DataFrame({'id':d['id'][m],'score':s[m]}).to_csv(out,index=False)
expected='3d6aacf25ac959ee0b2aa8f6fc45b4b705fad4a06cb50942434e9e3cb1407d06'
assert hashlib.sha256(out.read_bytes()).hexdigest()==expected
print('EXACT_REPLAY_OK',len(s),'scores; submission SHA256',expected)
