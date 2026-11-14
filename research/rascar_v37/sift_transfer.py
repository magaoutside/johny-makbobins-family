"""Classical RootSIFT matching of object texture; fit-day templates only.
Background keypoints never initiate a transfer. No retraining or test labels.
"""
from common37 import *
import cv2,time,joblib
cv2.setNumThreads(1)

def crop_features(r,sift):
    b=r[B].to_numpy(float);sz=np.maximum(b[2:]-b[:2],8);cc=(b[:2]+b[2:])/2
    lo=np.maximum(0,np.floor(cc-1.2*sz)).astype(int);hi=np.minimum([1280,720],np.ceil(cc+1.2*sz)).astype(int)
    raw=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720))
    gray=cv2.cvtColor(raw[lo[1]:hi[1],lo[0]:hi[0]],cv2.COLOR_BGR2GRAY);scale=384/max(gray.shape)
    gray=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    key,des=sift.detectAndCompute(gray,None)
    if des is None:return dict(xy=np.empty((0,2)),des=np.empty((0,128),np.float32),box=b,lo=lo,scale=scale)
    xy=np.array([k.pt for k in key],np.float32)/scale+lo
    des=np.sqrt(des/np.maximum(des.sum(1,keepdims=True),1e-6)).astype(np.float32)
    return dict(xy=xy,des=des,box=b,lo=lo,scale=scale)

def match(src,dst):
    inside=((src['xy']>=src['box'][:2])&(src['xy']<=src['box'][2:])).all(1)
    aa=src['xy'][inside];ad=src['des'][inside];bd=dst['des'];bb=dst['xy']
    if len(ad)<6 or len(bd)<6:return None
    bf=cv2.BFMatcher(cv2.NORM_L2);pairs=bf.knnMatch(ad,bd,k=2);reverse=bf.match(bd,ad)
    keep=[m for m,n in pairs if m.distance<.75*n.distance and reverse[m.trainIdx].trainIdx==m.queryIdx]
    if len(keep)<6:return None
    a=np.array([aa[m.queryIdx] for m in keep],np.float32);b=np.array([bb[m.trainIdx] for m in keep],np.float32)
    err=max(2.,.04*np.linalg.norm(dst['box'][2:]-dst['box'][:2]))
    mat,mask=cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=err,maxIters=2000,confidence=.99,refineIters=10)
    if mat is None:return None
    mask=mask.ravel().astype(bool);n=int(mask.sum())
    if n<6 or n/len(a)<.6:return None
    coverage=cv2.contourArea(cv2.convexHull(a[mask]))/np.prod(src['box'][2:]-src['box'][:2])
    sv=np.linalg.svd(mat[:,:2],compute_uv=False)
    if coverage<.2 or np.linalg.det(mat[:,:2])<=0 or sv.max()/sv.min()>2.5 or sv.min()<.3 or sv.max()>3.:return None
    x1,y1,x2,y2=src['box'];pts=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])@mat[:,:2].T+mat[:,2]
    ref=np.r_[pts.min(0),pts.max(0)]
    if np.any(ref<0) or np.any(ref>np.array([1280,720,1280,720])):return None
    return dict(ref=ref,inliers=n,coverage=float(coverage),similarity=float(1-np.mean([m.distance for m in keep])))

def main():
    protocol=dict(descriptor='RootSIFT',template_keypoints='inside official fit-day GT only',ratio=.75,min_inliers=6,min_coverage=.2,
        affine_RANSAC_error='max(2px,.04 query diagonal)',min_reference_frames=2,reference_agreement_iou=.65,
        source='https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html',test_labels_used=False)
    (A/'inputs/sift_protocol.json').write_text(json.dumps(protocol,indent=2))
    manual=h[h.class_name.isin(['cup','tray'])].copy();manual['id']=['sift_gt_'+str(i) for i in manual.index];manual['split']='manual'
    rec=pd.concat([manual,q[q.class_name.isin(['cup','tray'])].assign(split='oof'),te[te.class_name.isin(['cup','tray'])].assign(split='test')],ignore_index=True)
    sift=cv2.SIFT_create(nfeatures=1200,contrastThreshold=.015,edgeThreshold=12)
    feats={r.id:crop_features(r,sift) for _,r in rec.iterrows()};joblib.dump(feats,A/'features/sift_features.joblib');records=[];transfers=[]
    for fit in DAYS:
        bank=rec[rec.split.eq('manual')&rec.day.eq(fit)];evaluation=rec[rec.split.eq('test')|rec.day.ne(fit)]
        assert not set(bank.frame_id)&set(evaluation.frame_id)
        for _,r in evaluation.iterrows():
            candidates=[]
            for _,t in bank[bank.class_name.eq(r.class_name)].iterrows():
                found=match(feats[t.id],feats[r.id])
                if found is None:continue
                ref=found.pop('ref');z=dict(id=r.id,split=r.split,fit_day=fit,template_id=t.id,template_frame=t.frame_id,**found,**dict(zip(['ref_'+b for b in B],ref)));candidates.append(z);transfers.append(z)
            d=dict(id=r.id,split=r.split,frame_id=r.frame_id,day=r.day,class_name=r.class_name,fit_day=fit,support=0,ref_iou=0.,similarity=0.,valid=False)
            if candidates:
                cc=pd.DataFrame(candidates);boxes=cc[['ref_'+b for b in B]].to_numpy();ov=iou(boxes,boxes)
                counts=np.array([cc.loc[m>=.65,'template_frame'].nunique() for m in ov]);j=int(np.argmax(counts+.01*cc.inliers.to_numpy()))
                cluster=cc[ov[j]>=.65];support=cluster.template_frame.nunique();ref=np.median(cluster[['ref_'+b for b in B]].to_numpy(),axis=0)
                d.update(support=support,valid=support>=2,ref_iou=float(iou(r[B],ref)[0,0]),similarity=float(cluster.similarity.mean()),**dict(zip(['ref_'+b for b in B],ref)))
            records.append(d)
    p=pd.DataFrame(records);p.to_csv(A/'features/sift_correspondence.csv',index=False);pd.DataFrame(transfers).to_csv(A/'audit/sift_transfers.csv',index=False)
    m=p[p.split.eq('manual')];out=[]
    for cls,d in m.groupby('class_name'):
        valid=d[d.valid];out.append(dict(class_name=cls,n=len(d),predicted=len(valid),valid_iou060=int(valid.ref_iou.ge(.6).sum()),median_iou=float(valid.ref_iou.median()) if len(valid) else None))
    pd.DataFrame(out).to_csv(A/'results/sift_manual.csv',index=False);print(pd.DataFrame(out).to_string(index=False));print(p.groupby(['split','class_name']).valid.sum().to_string())

if __name__=='__main__':main()
