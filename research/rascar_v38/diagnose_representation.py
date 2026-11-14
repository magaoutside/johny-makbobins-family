"""Measure information lost by occupancy/connected-components using official GT."""
from common38 import *
from refit import decode
import cv2

def main():
    rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');y=np.load(ROOT/'rascar_v31/features/dense_targets.npy',mmap_mode='r');out=[]
    for i,r in rec[rec.split.eq('oof')].iterrows():
        z=decode(y[i].astype(np.float32),r);truth=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)]
        v=iou(r[B],truth[B]).ravel();true_max=float(v.max()) if len(v) else 0.
        ref=np.array([z.get('ref_'+b,np.nan) for b in B]);cb=r[['crop_'+b for b in B]].to_numpy(float)
        same=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy(float)
        centers=(same[:,:2]+same[:,2:])/2;inside=((centers>=ref[:2])&(centers<=ref[2:])).all(1) if np.isfinite(ref).all() else np.zeros(len(same),bool)
        overlap=0. if not np.isfinite(z['overlap']) else z['overlap'];official=int(q.set_index('id').loc[r.id,'error'])
        out.append(dict(id=r.id,frame_id=r.frame_id,day=r.day,class_name=r.class_name,error=official,true_max_iou=true_max,oracle_occupancy_iou=overlap,oracle_reference_gt_centers=int(inside.sum()),
          oracle_geometry_error=int(overlap<.6),max_gt_error=int(true_max<.6),oracle_disagrees=bool((overlap<.6)!=official),**dict(zip(['oracle_'+b for b in B],ref))))
    d=pd.DataFrame(out);d.to_csv(A/'audit/occupancy_oracle.csv',index=False)
    summary=dict(rows=len(d),oracle_disagreements=int(d.oracle_disagrees.sum()),max_iou_disagrees_with_official=int(d.max_gt_error.ne(d.error).sum()),
      oracle_wrongly_rejects_correct=int((d.oracle_disagrees&d.error.eq(0)).sum()),oracle_wrongly_accepts_error=int((d.oracle_disagrees&d.error.eq(1)).sum()),
      by_class=d[d.oracle_disagrees].class_name.value_counts().to_dict(),note='Oracle predicts exact GT occupancy on the 32x32 grid, then the current component/max-IoU decode. This tests representation/decoder limits, not learned model performance.')
    (A/'results/occupancy_oracle_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    print(d[d.oracle_disagrees][['id','class_name','error','true_max_iou','oracle_occupancy_iou','oracle_reference_gt_centers']].to_string(index=False))

if __name__=='__main__':main()
