"""Transfer train boxes through mutual DINO patch correspondences and RANSAC.

No new backbone or trained detection head. Independent fit-day images are
the only box templates. This is not an implementation of the cited paper.
"""
from common37 import *
import cv2,time,torch
cv2.setNumThreads(1);torch.set_num_threads(3)
CB=['crop_'+c for c in B]

def normbox(r):
    cb=r[CB].to_numpy(float);return (r[B].to_numpy(float)-np.r_[cb[:2],cb[:2]])/np.r_[cb[2:]-cb[:2],cb[2:]-cb[:2]]

def match(src,tgt,box,xy):
    similarity=src@tgt.T
    ab=similarity.argmax(1);ba=similarity.argmax(0)
    inside=((xy>=box[:2])&(xy<=box[2:])).all(1)
    good=inside&(ba[ab]==np.arange(len(src)))&(similarity[np.arange(len(src)),ab]>.45)
    ids=np.flatnonzero(good)
    if len(ids)<6:return None
    a=xy[ids].astype(np.float32);b=xy[ab[ids]].astype(np.float32)
    transform,inlier=cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=.065,maxIters=1000,confidence=.99,refineIters=10)
    if transform is None:return None
    mask=inlier.ravel().astype(bool);n=int(mask.sum())
    if n<6 or n/len(ids)<.6:return None
    sv=np.linalg.svd(transform[:,:2],compute_uv=False)
    if np.linalg.det(transform[:,:2])<=0 or sv.min()<.35 or sv.max()>2.8 or sv.max()/sv.min()>2.5:return None
    coverage=float(cv2.contourArea(cv2.convexHull(a[mask]))/np.prod(box[2:]-box[:2]))
    if coverage<.25:return None
    corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]])
    projected=corners@transform[:,:2].T+transform[:,2]
    ref=np.r_[projected.min(0),projected.max(0)]
    if np.any(ref<0) or np.any(ref>1) or np.min(ref[2:]-ref[:2])<.05:return None
    mean_sim=float(similarity[ids[mask],ab[ids[mask]]].mean())
    residual=float(np.linalg.norm(a[mask]@transform[:,:2].T+transform[:,2]-b[mask],axis=1).mean())
    return dict(ref=ref,inliers=n,coverage=coverage,similarity=mean_sim,residual=residual)

def main():
    protocol=dict(grid=16,top_templates=8,min_mutual_matches=6,min_inlier_fraction=.6,min_convex_hull_coverage=.25,
        ransac_error=.065,reference_cluster_iou=.65,min_distinct_template_frames=2,classes=['cup','tray','nude_hands'],
        purpose='Transfer box from visually matching object; geometric consistency is not proof of correct class.',
        source='https://github.com/ShirAmir/dino-vit-features',test_labels_used=False)
    (A/'inputs/correspondence_protocol.json').write_text(json.dumps(protocol,indent=2))
    rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');arr=np.load(ROOT/'rascar_v31/features/dense_grids.npy',mmap_mode='r')
    use=rec.class_name.isin(protocol['classes'])
    rr=rec[use].copy();features=torch.from_numpy(np.asarray(arr[rr.feature_index],np.float32))
    features=torch.nn.functional.avg_pool2d(features,2).flatten(2).transpose(1,2)
    features=torch.nn.functional.normalize(features,dim=2).numpy()
    lookup={v:j for j,v in enumerate(rr.id)};yy,xx=np.mgrid[:16,:16];xy=np.c_[(xx.ravel()+.5)/16,(yy.ravel()+.5)/16].astype(np.float32)
    descriptors={}
    for _,r in rr.iterrows():
        b=normbox(r);inside=((xy>=b[:2])&(xy<=b[2:])).all(1);desc=features[lookup[r.id],inside].mean(0)
        descriptors[r.id]=desc/max(np.linalg.norm(desc),1e-8)
    records=[];rawrecords=[];start=time.time()
    for fit in DAYS:
        bank=rr[rr.split.eq('manual')&rr.day.eq(fit)&rr.id.str.endswith('_0')]
        evaluation=rr[(rr.split.isin(['oof','manual'])&rr.day.ne(fit)&(~rr.split.eq('manual')|rr.id.str.endswith('_0')))|rr.split.eq('test')]
        assert not set(bank.frame_id)&set(evaluation.frame_id)
        for k,(_,r) in enumerate(evaluation.iterrows()):
            templates=bank[bank.class_name.eq(r.class_name)].copy()
            templates['similarity']=[float(descriptors[r.id]@descriptors[id]) for id in templates.id]
            templates=templates.sort_values('similarity',ascending=False).drop_duplicates('frame_id').head(8)
            candidates=[]
            for _,template in templates.iterrows():
                found=match(features[lookup[template.id]],features[lookup[r.id]],normbox(template),xy)
                if found is None:continue
                cb=r[CB].to_numpy(float);box=found.pop('ref')*np.r_[cb[2:]-cb[:2],cb[2:]-cb[:2]]+np.r_[cb[:2],cb[:2]]
                z=dict(id=r.id,split=r.split,frame_id=r.frame_id,class_name=r.class_name,fit_day=fit,template_id=template.id,template_frame=template.frame_id,**found,**dict(zip(['ref_'+b for b in B],box)))
                candidates.append(z);rawrecords.append(z)
            d=dict(id=r.id,split=r.split,frame_id=r.frame_id,day=r.day,class_name=r.class_name,fit_day=fit,support=0,ref_iou=0.,similarity=0.,valid=False)
            if candidates:
                cand=pd.DataFrame(candidates);boxes=cand[['ref_'+b for b in B]].to_numpy();ov=iou(boxes,boxes);counts=(ov>=.65).sum(1);j=int(np.argmax(counts+.01*cand.similarity.to_numpy()))
                cluster=cand[ov[j]>=.65];box=np.median(cluster[['ref_'+b for b in B]].to_numpy(),axis=0)
                d.update(support=len(cluster),valid=len(cluster)>=2,ref_iou=float(iou(r[B],box)[0,0]),similarity=float(cluster.similarity.mean()),**dict(zip(['ref_'+b for b in B],box)))
            records.append(d)
            if k%80==0:print('MATCH',fit,k,len(evaluation),round(time.time()-start,1),flush=True)
    pred=pd.DataFrame(records);pred.to_csv(A/'features/correspondence.csv',index=False);pd.DataFrame(rawrecords).to_csv(A/'audit/template_transfers.csv',index=False)
    # All manual targets are from the held-out day, not template fitting day.
    m=pred[pred.split.eq('manual')];out=[]
    for cls,d in m.groupby('class_name'):
        valid=d[d.valid];out.append(dict(class_name=cls,n=len(d),predicted=len(valid),valid_iou060=int(valid.ref_iou.ge(.6).sum()),median_iou=float(valid.ref_iou.median()) if len(valid) else None))
    pd.DataFrame(out).to_csv(A/'results/correspondence_manual.csv',index=False);print(pd.DataFrame(out).to_string(index=False))

if __name__=='__main__':main()
