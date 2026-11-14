"""Package small data/code only; all neural computation runs on Kaggle."""
from common56 import *
import shutil,zipfile,ast
G=A/'kaggle_payload';(G/'rascar_v56').mkdir(parents=True,exist_ok=True)
for p in ['inputs','features','models','results','audit','logs']:(G/'rascar_v56'/p).mkdir(exist_ok=True)
q,t,h=inputs();bo,bt=incumbent(q,t);vo,vt=baseline(q,t)
sys.path.insert(0,str(ROOT/'rascar_v55'));from train import predecessors
po,pt=predecessors(q,t)
q.assign(v52=bo,v31=vo,v22=po).to_csv(G/'rascar_v56/inputs/q.csv',index=False)
t.assign(v52=bt,v31=vt,v22=pt).to_csv(G/'rascar_v56/inputs/t.csv',index=False)
h.to_csv(G/'rascar_v56/inputs/h.csv',index=False)
for name in ['dense_records.csv']:
    dst=G/'rascar_v31/inputs'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'rascar_v31/inputs'/name,dst)
dst=G/'rascar_v31/features/dense_targets.npy';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'rascar_v31/features/dense_targets.npy',dst)
for family in ['base13','base23','anchor']:
    for fit in DAYS:
        name=f'{family}_fit{fit[-2:]}_bank.csv';shutil.copy2(A/'features'/name,G/'rascar_v56/features'/name)
shutil.copy2(ROOT/'rascar_v52/inputs/rows.csv',G/'rascar_v56/inputs/rows.csv')
for name in ['sam2_full.csv','sam2_crop.csv','v1_anchor_mean3.csv']:
    shutil.copy2(A/'features'/name,G/'rascar_v56/features'/name)
for p in (A/'features').glob('tabiclv1_anchor*_predictions.csv'):shutil.copy2(p,G/'rascar_v56/features'/p.name)
for p in (A/'models').glob('tabiclv1_anchor*.pkl'):shutil.copy2(p,G/'rascar_v56/models'/p.name)
for p in (A/'models').glob('tabiclv1_manifest*.json'):shutil.copy2(p,G/'rascar_v56/models'/p.name)
for name in ['dense_backbone.py','ensemble_audit.py']:
    code=(A/name).read_text()
    if name=='dense_backbone.py':
        code=code.replace('for mix in [0.,.5]:','for mix in [0.]:').replace("default='mps'","default='cuda:0'")
    (G/'rascar_v56'/name).write_text(code)
core=(ROOT/'rascar_v41/common.py').read_text();tree=ast.parse(core)
functions={node.name:ast.get_source_segment(core,node) for node in tree.body if isinstance(node,ast.FunctionDef)}
original=(A/'common56.py').read_text();tree=ast.parse(original);func56={node.name:ast.get_source_segment(original,node) for node in tree.body if isinstance(node,ast.FunctionDef)}
from inspect import getsource
stub='''from pathlib import Path
import os,sys,json,time,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score as AP
average_precision_score=AP
A=Path(__file__).resolve().parent;ROOT=A.parent;D=ROOT/'rascar_data_full'
B=['x1','y1','x2','y2'];CB=['crop_x1','crop_y1','crop_x2','crop_y2'];CL=['person','nude_hands','cup','tray'];DAYS=['2026-10-25','2026-10-26']
for p in ['inputs','features','models','results/oof','results/submissions','audit','logs']:(A/p).mkdir(parents=True,exist_ok=True)
def inputs():return tuple(pd.read_csv(A/f'inputs/{x}.csv') for x in ['q','t','h'])
def incumbent(q,t):return q.v52.to_numpy().copy(),t.v52.to_numpy().copy()
def baseline(q,t):return q.v31.to_numpy().copy(),t.v31.to_numpy().copy()
def predecessors(q,t):return q.v22.to_numpy().copy(),t.v22.to_numpy().copy()
'''
stub+='\n'+functions['iou']+'\n'+functions['metrics']+'\n'+getsource(score_dense)+'\n'+func56['save_score']+'\n'+func56['dense_variants']+'\n'
(G/'rascar_v56/common56.py').write_text(stub)
for name in ['kaggle_bootstrap.py','kaggle_tabular.py','kaggle_finish.py']:
    if (A/name).exists():shutil.copy2(A/name,G/'rascar_v56'/name)
for p in G.rglob('*.py'):ast.parse(p.read_text())
out=A/'kaggle_payload.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in sorted(G.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(G))
print('PAYLOAD',out.stat().st_size,'sha256',hashlib.sha256(out.read_bytes()).hexdigest())
