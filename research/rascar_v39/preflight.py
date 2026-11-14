"""Check set-target identity, loss invariance, truncation and exact matching."""
from set_detector import *

def assign(queries,refs):
    mat=iou(queries,refs);ok=mat>=.6;correct=np.zeros(len(queries),bool)
    if mat.size:
        ii,jj=linear_sum_assignment(np.where(ok,min(mat.shape)+1+mat,0),maximize=True)
        correct[ii[ok[ii,jj]]]=True
    return ~correct

def main():
    data=Data();rec=data.rec;pred=[];coverage=[]
    for (frame,cls),g in q.groupby(['frame_id','class_name']):
        truth=h[h.frame_id.eq(frame)&h.class_name.eq(cls)][B].to_numpy(float)
        # Set representation retains separate continuous boxes, including overlaps.
        cxcy=(truth[:,:2]+truth[:,2:])/2;wh=truth[:,2:]-truth[:,:2]
        recovered=xyxy(torch.tensor(np.c_[cxcy,wh],dtype=torch.float32)).numpy()
        labels=assign(g[B].to_numpy(),recovered)
        for ident,label in zip(g.id,labels):pred.append(dict(id=ident,recomputed_error=int(label)))
    prediction=pd.DataFrame(pred).set_index('id').loc[q.id].recomputed_error.to_numpy()
    for i,r in rec[rec.split.eq('oof')].iterrows():
        labels,boxes=data.targets[i];bb=xyxy(boxes).numpy();cb=r[CB].to_numpy(float)
        keep=labels.numpy()==CL.index(r.class_name)
        internal=np.array([cb[0]>0,cb[1]>0,cb[2]<1280,cb[3]<720])
        cropped=(np.c_[bb[:,:2]<.02,bb[:,2:]>.98]&internal).any(1);good=bb[keep&~cropped]
        good=good*np.r_[cb[2:]-cb[:2],cb[2:]-cb[:2]]+np.r_[cb[:2],cb[:2]]
        ov=iou(r[B],good).ravel();coverage.append(dict(id=r.id,frame_id=r.frame_id,class_name=r.class_name,reference_count=len(good),best_iou=float(ov.max()) if len(ov) else np.nan))
    d=pd.DataFrame(coverage).set_index('id').loc[q.id];d['error']=q.error.to_numpy();d.to_csv(A/'audit/oracle_crop_coverage.csv')
    # Check Hungarian loss under target permutation and overlapping same-class objects.
    target=torch.tensor([[.48,.48,.4,.4],[.52,.52,.4,.4]],dtype=torch.float32);lab=torch.tensor([1,1])
    logits=torch.full((1,32,5),-15.);logits[:,:,4]=15.;logits[0,:2,4]=-15.;logits[0,:2,1]=15.
    boxes=torch.ones(1,32,4)*.2;boxes[0,:2]=target
    loss1=float(set_loss([(logits,boxes)],[(lab,target)]));loss2=float(set_loss([(logits,boxes)],[(lab.flip(0),target.flip(0))]))
    example=rec[rec.split.eq('oof')].iloc[0];decoded=decode(logits[0],boxes[0],example)
    result=dict(official_oof_rows=len(q),native_set_matching_disagreements=int((prediction!=q.error.to_numpy()).sum()),
      previous_union_occupancy_disagreements=34,overlap_pair_iou=float(iou(xyxy(target).numpy()[0],xyxy(target).numpy()[1])[0,0]),overlap_pair_survives_without_nms=len(decoded)==2,
      exact_set_loss=loss1,target_permutation_loss_difference=abs(loss1-loss2),maximum_targets_in_crop=max(len(t[0]) for t in data.targets.values()),slots=32,
      correct_queries_with_complete_gt_in_own_crop=int((d.error.eq(0)&d.best_iou.ge(.6)).sum()),correct_queries_total=int(q.error.eq(0).sum()),
      test_target_count=int(sum(rec.loc[i,'split']=='test' for i in data.targets)),
      warning='Oracle checks representation only; actual learned prediction quality is still unknown. Internal crop boundaries must abstain.')
    (A/'audit/preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    assert result['native_set_matching_disagreements']==0 and result['target_permutation_loss_difference']<1e-7 and result['overlap_pair_survives_without_nms'] and result['test_target_count']==0

if __name__=='__main__':main()
