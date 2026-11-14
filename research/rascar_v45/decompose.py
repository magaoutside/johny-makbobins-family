"""Decompose merged V31 occupancy using independent instance anchors.

No training, query-conditioned fitting, manual test labels, or ID overrides.
Unlike the earlier BPJ replacement, infer rectangles from the original dense map.
"""
from common45 import *
import cv2,time
cv2.setNumThreads(1)

def coverage(boxes):
    boxes=np.asarray(boxes).reshape(-1,4)
    xx=np.arange(32);yy=np.arange(32)
    wx=np.maximum(0,np.minimum(xx[None]+1,boxes[:,2,None])-np.maximum(xx[None],boxes[:,0,None]))
    wy=np.maximum(0,np.minimum(yy[None]+1,boxes[:,3,None])-np.maximum(yy[None],boxes[:,1,None]))
    return wy[:,:,None]*wx[:,None,:]

def fit_rectangles(prob,mask,initial,points,prior):
    boxes=np.clip(initial,0,32).copy();initial=boxes.copy()
    wh=np.maximum(initial[:,2:]-initial[:,:2],2);den=np.tile(wh,(1,2))
    # Partial-pixel rectangle occupancy preserves subcell edges. Prior controls
    # changes to the external detector; it is not supervision from the query.
    region=cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    target=prob*mask
    def loss(bb):
        u=coverage(bb).max(0)
        return np.mean((u[region]-target[region])**2)+prior*np.mean(((bb-initial)/den)**2)
    best=loss(boxes)
    for step in [2.,1.,.5]:
        for iteration in range(3):
            changed=False
            for j in range(len(boxes)):
                for side in range(4):
                    bestbox=None
                    for sign in [-1,1]:
                        bb=boxes.copy();bb[j,side]+=sign*step
                        if (bb[j,:2]<0).any() or (bb[j,2:]>32).any() or (bb[j,2:]-bb[j,:2]<1.5).any():continue
                        if (np.abs((bb[j]-initial[j])/den[j])>.65).any():continue
                        if not ((points[j]>=bb[j,:2])&(points[j]<=bb[j,2:])).all():continue
                        z=loss(bb)
                        if z<best-1e-8:best=z;bestbox=bb
                    if bestbox is not None:boxes=bestbox;changed=True
            if not changed:break
    return boxes,best

def split_map(prob,cb,boxes,points,method):
    size=cb[2:]-cb[:2];gridbox=(boxes-np.tile(cb[:2],2))/np.tile(size,2)*32
    gp=(points-cb[:2])/size*32
    n,labels,stats,_=cv2.connectedComponentsWithStats((prob>=.5).astype(np.uint8),8)
    results=[]
    for j in range(1,n):
        x,y,w,hh,ar=stats[j]
        if ar<4:continue
        old=np.array([x,y,x+w,y+hh],float);mask=labels==j
        # An anchor must lie in the same observed component, with one-cell tolerance.
        padded=cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8))
        ix=np.floor(gp).astype(int);inside=((ix>=0)&(ix<32)).all(1)
        selected=np.array([k for k in np.flatnonzero(inside) if padded[ix[k,1],ix[k,0]]],int)
        if len(selected)<2 or len(selected)>5:
            results.append((old,False,0,float(prob[mask].mean())));continue
        bb=gridbox[selected];pp=gp[selected]
        if method=='external_control':new=bb
        elif method=='voronoi':
            yy,xx=np.mgrid[:32,:32];xy=np.stack([xx+.5,yy+.5],-1)
            center=(bb[:,:2]+bb[:,2:])/2;scale=np.maximum(bb[:,2:]-bb[:,:2],2)
            distance=(((xy[None]-center[:,None,None])/scale[:,None,None])**2).sum(-1)
            assigned=distance.argmin(0);new=[]
            for k in range(len(bb)):
                ys,xs=np.where(mask&(assigned==k))
                if len(xs)<4:continue
                new.append([xs.min(),ys.min(),xs.max()+1,ys.max()+1])
            new=np.array(new).reshape(-1,4)
        else:new,_=fit_rectangles(prob,mask,bb,pp,.02 if method=='rect_weak' else .20)
        if len(new)<2:
            results.append((old,False,len(selected),float(prob[mask].mean())));continue
        for b in new:results.append((b,True,len(selected),float(prob[mask].mean())))
    return [(b/32*np.tile(size,2)+np.tile(cb[:2],2),changed,count,conf) for b,changed,count,conf in results]

def main():
    protocol=dict(baseline='V31',methods=['external_control','voronoi','rect_weak','rect_strong'],classes=['person','nude_hands'],threshold=.5,anchor_conf=.3,anchor_nms_iou=.65,query_used_for_fitting=False,fit_day_rule='Only held-day original V31 maps for official OOF; two original fit-day maps for test',test_labels=False,search='Four fixed decoders, weights .10/.25, class scopes person/hands/both; repeated development OOF, no claim of untouched evaluation',anchor_sources=['BPJDet head','BPJDet hand local'],new_weights_downloaded=False)
    (A/'inputs/decomposition_protocol.json').write_text(json.dumps(protocol,indent=2))
    q,te,h=inputs();heads=load_jsonl('head');hands=load_jsonl('hand_local');rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv').set_index('id')
    out=[];start=time.time()
    for fit in DAYS:
        data=np.load(ROOT/f'rascar_v31/features/linear_mask_fit{fit[-2:]}_maps.npz');lookup={s:i for i,s in enumerate(data['ids'])}
        rows=pd.concat([q[q.day.ne(fit)].assign(split='oof'),te.assign(split='test')],ignore_index=True)
        for no,(_,r) in enumerate(rows.iterrows()):
            if r.class_name not in ['person','nude_hands']:continue
            prob=data['maps'][lookup[r.id],CL.index(r.class_name)].astype(float)
            cb=rec.loc[r.id,CB].to_numpy(float);bb,pp,_=anchors(r.frame_id,r.class_name,heads,hands)
            for method in protocol['methods']:
                refs=split_map(prob,cb,bb,pp,method)
                if not refs:continue
                overlaps=iou(r[B],np.array([x[0] for x in refs]))[0];chosen=int(overlaps.argmax());box,changed,count,confidence=refs[chosen]
                out.append(dict(id=r.id,frame_id=r.frame_id,day=r.day,split=r.split,fit_day=fit,class_name=r.class_name,method=method,overlap=overlaps[chosen],modified=changed,anchors=count,confidence=confidence,**{'ref_'+b:v for b,v in zip(B,box)}))
            if no%100==0:print('DECODE',fit,no,round(time.time()-start,1),flush=True)
    pd.DataFrame(out).to_csv(A/'features/decomposition.csv',index=False)
    print('DONE',len(out),time.time()-start,flush=True)
if __name__=='__main__':main()
