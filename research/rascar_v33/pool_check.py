"""Previously reviewed positives, held by day; diagnostic, not an AP dataset."""
from boundary_head import *

def main():
    rec=pd.read_csv(ROOT/'rascar_v32/inputs/pool_control_records.csv');rec['split']='pool'
    feat=np.load(ROOT/'rascar_v32/features/pool_control_grids.npy',mmap_mode='r');records=[];summary=[]
    for fit in DAYS:
        ids=rec.index[rec.day.ne(fit)].to_numpy();rr=rec.loc[ids]
        c=torch.from_numpy(np.stack([cond(normbox(r[B].to_numpy(float),r[CB].to_numpy(float)),CL.index(r.class_name)) for _,r in rr.iterrows()]))
        for kind in ['manual','hardneg','regional_hardneg','pool_regional','pool_regional_balanced','pool_regional_balanced_rare']:
            model=RegionalHead() if 'regional' in kind else SideHead();model.load_state_dict(torch.load(A/f'models/side_{kind}_fit{fit[-2:]}.pt',weights_only=True));model.eval()
            with torch.inference_mode():out=model(torch.from_numpy(np.array(feat[ids],dtype=np.float32)),c)
            rows,_=decode(out,rr)
            for r in rows:records.append(dict(model=kind,fit_day=fit,**r))
    df=pd.DataFrame(records);df.to_csv(A/'audit/pool_boundary_predictions.csv',index=False)
    for (kind,cls),d in df.groupby(['model','class_name']):
        summary.append(dict(model=kind,class_name=cls,n=len(d),mean_iou=float(d.ref_iou.mean()),recall_iou06=float(d.ref_iou.ge(.6).mean()),mean_presence=float(d.presence.mean()),mean_predicted_quality=float(d.predicted_quality.mean())))
    pd.DataFrame(summary).to_csv(A/'results/pool_controls.csv',index=False);print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
