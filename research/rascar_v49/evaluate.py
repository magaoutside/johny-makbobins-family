from pathlib import Path
import sys,argparse
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v46'))
from common46 import inputs,baseline,metrics,v22base,compose
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=4901);args=p.parse_args();suffix='' if args.seed==4901 else f'_s{args.seed}'
    q,te,h=inputs();bo,bt=baseline(q,te);bq,bte=v22base(q,te);f=pd.read_csv(A/f'features/evidence{suffix}.csv');out=[]
    for (key,kind),g in f.groupby(['key','kind']):
        o=g[g.split.eq('oof')].set_index('id').reindex(q.id);t=[z.set_index('id').reindex(te.id) for _,z in g[g.split.eq('test')].groupby('fit_day')];s=compose(bq,o.overlap.to_numpy());tt=compose(bte,np.stack([z.overlap for z in t]));om=q.class_name.eq('tray').to_numpy();tm=te.class_name.eq('tray').to_numpy()
        for weight in [.25,.5,1.]:
            so=bo.copy();st=bt.copy();so[om]+=weight*(s[om]-bo[om]);st[tm]+=weight*(tt[tm]-bt[tm]);name=f'{key}_{kind}_tray_mix{int(weight*100)}{suffix}';q.assign(score=so).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=st).to_csv(A/f'results/submissions/{name}.csv',index=False);out.append(dict(name=name,**metrics(q,so),changed_test=int((abs(st-bt)>1e-9).sum())))
    d=pd.DataFrame(out).sort_values('ap',ascending=False);d.to_csv(A/f'results/experiments{suffix}.csv',index=False);print(d.to_string(index=False))
    p=pd.read_csv(A/f'audit/pool_geometry{suffix}.csv');p=p[p.role.eq('reserved')];r=p.groupby(['key','kind']).agg(frames=('frame_id','nunique'),mean_reference_iou=('reference_gt_iou','mean'),recall06=('reference_gt_iou',lambda x:float(x.ge(.6).mean()))).reset_index();r.to_csv(A/f'results/reserved_geometry{suffix}.csv',index=False);print(r.to_string(index=False))
if __name__=='__main__':main()
