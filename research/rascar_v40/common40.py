from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'rascar_v39'))
from common39 import B,CL,DAYS,q,te,h,BASE,iou,AP,np,pd
A=ROOT/'rascar_v40'
for name in ['inputs','features','models','results/oof','results/submissions','audit','logs']:(A/name).mkdir(parents=True,exist_ok=True)

def metric(rows,score):
    score=np.asarray(score);out={'ap':float(AP(rows.error,score))}
    for col in ['day','class_name']:
        for key,part in rows.groupby(col):out[key]=float(AP(part.error,score[rows[col].eq(key)]))
    return out

def save(name,scores,description):
    result=dict(experiment=name,**metric(q,scores['oof']),description=description,submitted=False,public_ap=None);changes=[]
    for split,rows in [('oof',q),('test',te)]:
        s=np.asarray(scores[split]);assert len(s)==len(rows) and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
        path=A/(f'results/oof/{name}.csv' if split=='oof' else f'results/submissions/submission_{name}.csv')
        (rows.assign(score=s) if split=='oof' else rows[['id']].assign(score=s)).to_csv(path,index=False)
        delta=s-BASE[split];changes.append(rows.assign(split=split,v31_score=BASE[split],new_score=s,delta=delta)[abs(delta)>1e-10]);result['changed_'+split]=int((abs(delta)>1e-10).sum())
    pd.concat(changes).to_csv(A/f'audit/{name}_changes.csv',index=False)
    print('RESULT',name,round(result['ap'],8),result['changed_oof'],result['changed_test'],flush=True)
    return result
