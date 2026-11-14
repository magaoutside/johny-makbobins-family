"""Opposite-day check on the 30 preserved reviewed pool positives."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v32'
sys.path.insert(0,str(A))
from conditional_masks import ConditionalMask,normalized_box,condition,decode,torch,np,pd,CL,B,DAYS,iou

def main():
    rows=pd.read_csv(A/'inputs/pool_control_records.csv');x=np.load(A/'features/pool_control_grids.npy');out=[]
    for fit in DAYS:
        ids=rows[rows.day.ne(fit)].index.to_numpy()
        c=np.stack([condition(normalized_box(rows.loc[i]),CL.index(rows.loc[i].class_name)) for i in ids])
        for kind in ['conditional_box','conditional_projection']:
            model=ConditionalMask();model.load_state_dict(torch.load(A/f'models/{kind}_fit{fit[-2:]}.pt',weights_only=True));model.eval()
            with torch.inference_mode():prob=model(torch.from_numpy(x[ids].astype(np.float32)),torch.from_numpy(c)).sigmoid().numpy()
            for j,i in enumerate(ids):
                r=rows.loc[i];bs=pd.DataFrame(decode(prob[j],r),columns=['class_name','threshold','confidence']+B)
                for cut in [.35,.5,.65]:
                    z=bs[bs.threshold.eq(cut)];ov=iou(r[B],z[B]).ravel();k=ov.argmax() if len(ov) else -1
                    out.append(dict(id=r.id,frame_id=r.frame_id,held_day=r.day,fit_day=fit,class_name=r.class_name,head=kind,cut=cut,overlap=float(ov[k]) if k>=0 else np.nan,confidence=float(z.iloc[k].confidence) if k>=0 else 0,**{f'ref_{b}':float(z.iloc[k][b]) if k>=0 else np.nan for b in B}))
    out=pd.DataFrame(out);out.to_csv(A/'audit/conditional_pool_evidence.csv',index=False)
    summary=out.groupby(['head','cut','class_name']).agg(n=('id','size'),coverage=('overlap',lambda x:x.notna().mean()),mean_iou=('overlap',lambda x:x.fillna(0).mean()),fraction_iou06=('overlap',lambda x:x.ge(.6).mean()))
    summary.to_csv(A/'results/conditional_pool_summary.csv');print(summary.to_string())
if __name__=='__main__':main()
