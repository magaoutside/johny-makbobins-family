from train import *
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score as AP
def iou_bounds(q,b,u):
    inner=b+np.array([u,u,-u,-u]);outer=b+np.array([-u,-u,u,u])
    inter=lambda x:np.maximum(0,np.minimum(q[2:],x[2:])-np.maximum(q[:2],x[:2])).prod()
    aq=np.maximum(q[2:]-q[:2],0).prod();ai=np.maximum(inner[2:]-inner[:2],0).prod();ao=np.maximum(outer[2:]-outer[:2],0).prod()
    return inter(inner)/(aq+ao-inter(outer)+1e-9),min(1.,inter(outer)/(aq+ai-inter(inner)+1e-9))
def main():
    torch.set_num_threads(3);rec=pd.read_csv(A/'inputs/pool_records.csv');rec=rec[rec.split.eq('reserved_holdout')];z=np.load(A/'features/pool_grids.npy',mmap_mode='r')
    labels=pd.read_csv(A/'pool_review/boxes.csv');out=[]
    for fit in DAYS:
      old=Head();w=torch.load(ROOT/f'rascar_v31/models/linear_mask_fit{fit[-2:]}.pt',map_location='cpu',weights_only=True)
      with torch.no_grad():old.net.weight.copy_(w['net.weight'][1:2]);old.net.bias.copy_(w['net.bias'][1:2])
      for variant in ['original','control','silver']:
        m=old if variant=='original' else Head()
        if variant!='original':m.load_state_dict(torch.load(A/f'models/{variant}_fit{fit[-2:]}.pt',weights_only=True))
        m.eval()
        with torch.inference_mode():maps=m(torch.tensor(np.asarray(z[rec.index]),dtype=torch.float32)).sigmoid().numpy()
        for k,(_,r) in enumerate(rec.iterrows()):
          gt=labels[labels.review_index.eq(r.review_index)];qb=r[B].to_numpy(float);bb=gt[B].to_numpy(float);cb=r[CB].to_numpy(float);sz=cb[2:]-cb[:2]
          ov=float(iou(qb,bb).max());bounds=np.array([iou_bounds(qb,g,r.uncertainty) for g in bb]);lo=bounds[:,0].max();hi=bounds[:,1].max()
          n,_,stats,_=cv2.connectedComponentsWithStats((maps[k]>=.5).astype(np.uint8),8);cand=[]
          for j in range(1,n):
            x,y,ww,hh,area=stats[j]
            if area<4:continue
            ref=np.r_[x,y,x+ww,y+hh]/32*np.tile(sz,2)+np.tile(cb[:2],2);cand.append((float(iou(qb,ref)[0,0]),ref))
          if cand:qi,ref=max(cand,key=lambda x:x[0]);ri=float(iou(ref,bb).max());score=float(np.clip((.85-qi)/.5,0,1))
          else:qi=np.nan;ri=0;score=np.nan
          target=0 if lo>=.6 else 1 if hi<.6 else np.nan
          out.append(dict(id=r.id,frame_id=r.frame_id,day=r.day,review_index=r.review_index,variant=variant,fit_day=fit,silver_query_iou=ov,robust_iou_lower=lo,robust_iou_upper=hi,robust_geometry_error=target,reference_gt_iou=ri,query_reference_iou=qi,geometry_score=score))
    df=pd.DataFrame(out);df.to_csv(A/'audit/reserved_pool_geometry.csv',index=False)
    summary=[]
    for variant,g in df.groupby('variant'):
        by=g.groupby('id').mean(numeric_only=True);valid=by.robust_geometry_error.notna()&by.geometry_score.notna();ap=AP(by.loc[valid,'robust_geometry_error'],by.loc[valid,'geometry_score']) if by.loc[valid,'robust_geometry_error'].nunique()==2 else None
        summary.append(dict(variant=variant,frames=len(by),mean_reference_iou=float(by.reference_gt_iou.mean()),robust_labelled=int(valid.sum()),geometry_ap=ap,warning='Unofficial six-frame silver geometry panel, not full error-ranking labels or leaderboard proxy; models evaluated on same 128px crop'))
    (A/'audit/reserved_pool_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
