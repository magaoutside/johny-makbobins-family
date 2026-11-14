from common45 import *
q,te,h=inputs();d=pd.read_csv(A/'features/decomposition.csv');d=d[d.split.eq('oof')];old=pd.read_csv(ROOT/'rascar_v31/features/linear_mask_cut0.5_oof_evidence.csv').set_index('id');q=q.set_index('id');rows=[]
for _,r in d[d.modified.eq(True)].iterrows():
    qr=q.loc[r.id];gt=h[h.frame_id.eq(qr.frame_id)&h.class_name.eq(qr.class_name)][B].to_numpy(float)
    if not len(gt):continue
    g=gt[iou(qr[B],gt)[0].argmax()];newbox=r[['ref_'+b for b in B]].to_numpy(float);oldbox=old.loc[r.id,['ref_'+b for b in B]].to_numpy(float)
    before=float(iou(oldbox,g)[0,0]);after=float(iou(newbox,g)[0,0])
    rows.append(dict(id=r.id,frame_id=r.frame_id,method=r.method,class_name=r.class_name,error=qr.error,old_reference_gt_iou=before,new_reference_gt_iou=after,delta=after-before,query_gt_iou=float(iou(qr[B],g)[0,0]),old_query_iou=old.loc[r.id,'overlap'],new_query_iou=r.overlap))
out=pd.DataFrame(rows);out.to_csv(A/'audit/geometry_quality.csv',index=False)
summary=out.groupby(['method','class_name']).agg(n=('id','size'),mean_old=('old_reference_gt_iou','mean'),mean_new=('new_reference_gt_iou','mean'),improved=('delta',lambda x:int((x>1e-6).sum())),worsened=('delta',lambda x:int((x<-1e-6).sum())))
summary.to_csv(A/'audit/geometry_summary.csv');print(summary.to_string())
