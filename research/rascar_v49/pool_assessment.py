"""Uncertainty-aware diagnostics of reserved silver boxes, no AP claim."""
from train_tray import *
def main():
    truth=pd.read_csv(A/'pool_review/boxes.csv').set_index('frame_id');out=[]
    for path in sorted((A/'audit').glob('pool_geometry*.csv')):
        p=pd.read_csv(path);p=p[p.role.eq('reserved')];seed=4901 if path.stem=='pool_geometry' else int(path.stem.split('_s')[-1])
        for _,r in p.iterrows():
            b=truth.loc[r.frame_id,B].to_numpy(float);tol=float(truth.loc[r.frame_id,'edge_uncertainty_px']);ref=r[['ref_'+c for c in B]].to_numpy(float);old=p[p.key.eq(r.key)&p.kind.eq('original')&p.frame_id.eq(r.frame_id)].iloc[0][['ref_'+c for c in B]].to_numpy(float)
            vals=[[v-tol,v+tol,np.clip(rr,v-tol,v+tol),np.clip(oo,v-tol,v+tol)] for v,rr,oo in zip(b,ref,old)];choices=np.array(list(itertools.product(*vals)));choices=choices[np.isfinite(choices).all(1)&(choices[:,2:]>choices[:,:2]).all(1)]
            if not len(choices):choices=np.array(list(itertools.product(*[[v-tol,v+tol] for v in b])))
            new_iou=iou(ref,choices)[0] if np.isfinite(ref).all() else np.zeros(len(choices));old_iou=iou(old,choices)[0] if np.isfinite(old).all() else np.zeros(len(choices));delta=new_iou-old_iou
            out.append(dict(seed=seed,key=r.key,kind=r.kind,frame_id=r.frame_id,review_index=r.review_index,nominal_iou=r.reference_gt_iou,iou_low=float(new_iou.min()),iou_high=float(new_iou.max()),min_gain=float(delta.min()),max_gain=float(delta.max()),robust_query_label=r.robust_label))
    d=pd.DataFrame(out);d.to_csv(A/'audit/reserved_uncertainty.csv',index=False);s=d.groupby(['seed','key','kind']).agg(n=('frame_id','nunique'),nominal_mean=('nominal_iou','mean'),lower_mean=('iou_low','mean'),nominal_recall06=('nominal_iou',lambda x:float(x.ge(.6).mean())),robust_recall06=('iou_low',lambda x:float(x.ge(.6).mean())),robust_better_than_original=('min_gain',lambda x:int(x.gt(1e-9).sum())),robust_worse_than_original=('max_gain',lambda x:int(x.lt(-1e-9).sum()))).reset_index();s.to_csv(A/'results/reserved_uncertainty_summary.csv',index=False);print(s.to_string(index=False))
if __name__=='__main__':main()
