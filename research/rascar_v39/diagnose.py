"""Compare learned references to official held-day geometry, OOF only."""
from common39 import *

def main():
    ev=pd.read_csv(A/'features/set_long_identity_matching_evidence.csv');out=[]
    prediction=pd.read_csv(A/'results/oof/v39_set_long_identity_geometry_all_RESEARCH.csv').set_index('id')
    for i,r in q.iterrows():
        gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy(float)
        d=ev[ev.split.eq('oof')&ev.id.eq(r.id)];ref=d[['ref_'+b for b in B]].to_numpy(float);valid=np.isfinite(ref).all(1);ref=ref[valid]
        true_query=float(iou(r[B],gt).max()) if len(gt) else 0.
        agreement=float(iou(ref,ref).min()) if len(ref)==3 else np.nan
        ref_truth=iou(ref,gt).max(1) if len(gt) and len(ref) else np.zeros(len(ref))
        inferred=d.overlap.to_numpy();score=float(prediction.loc[r.id,'score'])
        out.append(dict(id=r.id,frame_id=r.frame_id,class_name=r.class_name,day=r.day,official_error=int(r.error),query_max_iou_gt=true_query,reference_count=len(ref),reference_seed_agreement=agreement,reference_min_iou_gt=float(ref_truth.min()) if len(ref_truth) else np.nan,reference_mean_iou_gt=float(ref_truth.mean()) if len(ref_truth) else np.nan,predicted_query_iou_min=float(np.nanmin(inferred)) if np.isfinite(inferred).any() else np.nan,predicted_query_iou_max=float(np.nanmax(inferred)) if np.isfinite(inferred).any() else np.nan,v31_score=BASE['oof'][i],new_score=score,delta=score-BASE['oof'][i]))
    d=pd.DataFrame(out);d.to_csv(A/'audit/official_reference_diagnostic.csv',index=False)
    d[abs(d.delta)>1e-10].to_csv(A/'audit/official_changed_reference_diagnostic.csv',index=False)
    print(d[abs(d.delta)>1e-10][['class_name','official_error','query_max_iou_gt','reference_seed_agreement','reference_min_iou_gt','delta']].to_string(index=False))

if __name__=='__main__':main()
