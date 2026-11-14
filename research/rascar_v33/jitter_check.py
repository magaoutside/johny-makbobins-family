"""Synthetic held-day boundary stress test, separate from official OOF AP."""
from boundary_head import *
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr

def main():
    data=Data();rec=data.rec;records=[]
    for fit in DAYS:
        held=rec[rec.split.eq('manual')&rec.day.ne(fit)&rec.class_name.isin(CL)].drop_duplicates('origin_id')
        rows=[]
        for i,r in held.iterrows():
            b=r[B].to_numpy(float);wh=b[2:]-b[:2];center=(b[:2]+b[2:])/2
            for kind,box in [('original',b),('shrink',np.r_[center-.35*wh,center+.35*wh]),('expand',np.r_[center-.7*wh,center+.7*wh]),('shift',b+np.tile([.3*wh[0],0],2)),('bottom_clip',b-np.array([0,0,0,.45*wh[1]]))]:
                z=r.copy();z[B]=box;z['aug']=kind;z['gt_iou']=float(iou(b,box)[0,0]);z['index']=i;rows.append(z)
        rr=pd.DataFrame(rows)
        for kind in ['manual','hardneg','regional_hardneg','pool_regional','pool_regional_balanced','pool_regional_balanced_rare']:
            model=RegionalHead() if 'regional' in kind else SideHead();model.load_state_dict(torch.load(A/f'models/side_{kind}_fit{fit[-2:]}.pt',weights_only=True));model.eval()
            for off in range(0,len(rr),16):
                sub=rr.iloc[off:off+16];c=torch.from_numpy(np.stack([cond(normbox(r[B].to_numpy(float),r[CB].to_numpy(float)),CL.index(r.class_name)) for _,r in sub.iterrows()]))
                with torch.inference_mode():out=model(data.features(sub['index'].to_numpy(int)),c)
                decoded,_=decode(out,sub)
                for pred,(_,r) in zip(decoded,sub.iterrows()):records.append(dict(model=kind,fit_day=fit,aug=r.aug,gt_iou=r.gt_iou,**pred))
    df=pd.DataFrame(records);df.to_csv(A/'audit/synthetic_boundary_stress.csv',index=False);summary=[]
    for (kind,cls),d in df.groupby(['model','class_name']):
        y=d.gt_iou.lt(.6).astype(int)
        summary.append(dict(model=kind,class_name=cls,n=len(d),ap_quality=float(average_precision_score(y,1-d.predicted_quality)),ap_decoded_iou=float(average_precision_score(y,1-d.ref_iou)),quality_mae=float(abs(d.predicted_quality-d.gt_iou).mean()) if 'regional' not in kind else None,quality_spearman=float(spearmanr(d.gt_iou,d.predicted_quality).statistic)))
    pd.DataFrame(summary).to_csv(A/'results/synthetic_boundary_stress.csv',index=False);print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
