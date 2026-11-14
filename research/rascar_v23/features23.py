"""Classical image evidence and relations between objects; no query labels used.

Fitted color statistics, empty-scene backgrounds and body templates use only
the fit day. A training frame is excluded from the background/template bank.
"""
from common23 import *
from functools import lru_cache
import cv2
cv2.setNumThreads(2)

heads=pd.read_csv(ROOT/'rascar_v16/wevis_heads_references.csv')
poses=pd.read_csv(ROOT/'rascar_v16/multiview_pose.csv')
refs=pd.concat([pd.read_csv(ROOT/f'rascar_v15/v8/fused_{split}_references.csv') for split in ['train','test']],ignore_index=True)
pool=pd.read_csv(V22/'inputs/reviewed_pool_hands.csv')
paths={p.stem:p for folder in [ROOT/'rascar_v16/images',ROOT/'rascar_v16/test_images'] for p in folder.glob('*.jpg')}

@lru_cache(maxsize=20)
def image(fid):
    im=cv2.imread(str(paths[fid]));assert im is not None
    return cv2.resize(im,(1280,720))

def clipbox(b,w=1280,hh=720):
    b=np.asarray(b,float);lo=np.maximum(0,np.floor(b[:2])).astype(int);hi=np.minimum([w,hh],np.ceil(b[2:])).astype(int)
    hi=np.maximum(hi,lo+1);return np.r_[lo,hi]
def crop(im,b):
    b=clipbox(b,im.shape[1],im.shape[0]);return im[b[1]:b[3],b[0]:b[2]]
def nms(d,cut=.5):
    keep=[]
    for ix,r in d.sort_values('confidence',ascending=False).iterrows():
        if not keep or iou(r[B],d.loc[keep,B]).max()<cut:keep.append(ix)
    return d.loc[keep]

@lru_cache(None)
def frame_objects(fid):
    rr=refs[refs.frame_id.eq(fid)&refs.confidence.ge(.45)]
    return {c:nms(rr[rr.class_name.eq(c)]) for c in ['table','person','nude_hands','cup','tray']}

@lru_cache(None)
def frame_pose(fid):
    out=[]
    for r in poses[poses.frame_id.eq(fid)&poses.confidence.ge(.3)].itertuples():
        kp=np.array([[getattr(r,f'kp{k}_x'),getattr(r,f'kp{k}_y'),getattr(r,f'kp{k}_conf')] for k in range(17)])
        out.append((r,kp))
    return out

@lru_cache(None)
def skin_hist(fit):
    pos=np.ones((32,32),float)*.01;neg=np.ones((32,32),float)*.01
    for r in h[h.day.eq(fit)&h.class_name.eq('nude_hands')].itertuples():
        b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];im=image(r.frame_id)
        for arr,hist in [(crop(im,np.r_[b[:2]+.18*wh,b[2:]-.18*wh]),pos),
                         (crop(im,np.r_[b[:2]-.5*wh,b[2:]+.5*wh]),neg)]:
            lab=cv2.cvtColor(arr,cv2.COLOR_BGR2LAB);xy=np.minimum(lab[:,:,1:].reshape(-1,2)//8,31).astype(np.int32)
            counts=np.bincount(xy[:,0]*32+xy[:,1],minlength=1024).reshape(32,32)
            hist+=counts/max(counts.sum(),1)
    # Normalize per-crop contributions; histogram ratio is soft evidence only.
    pos=cv2.GaussianBlur(pos,(3,3),.7);neg=cv2.GaussianBlur(neg,(3,3),.7)
    pos/=pos.sum();neg/=neg.sum();return pos/(pos+neg)

@lru_cache(maxsize=20)
def pixel_maps(fid,fit):
    im=image(fid);lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    idx=np.minimum(lab[:,:,1:]//8,31);p=skin_hist(fit)[idx[:,:,0],idx[:,:,1]]
    return lab,gray,p

@lru_cache(None)
def empty_bank(fit):
    allids=sorted(h[h.day.eq(fit)].frame_id.unique());occupied=set(h[h.day.eq(fit)&h.class_name.eq('person')].frame_id)
    out=[]
    for fid in allids:
        if fid in occupied:continue
        im=cv2.resize(image(fid),(640,360));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        out.append((fid,im,float(gray.mean()),float(np.mean(np.std(im.astype(float),axis=2)))))
    return out

@lru_cache(maxsize=20)
def foreground(fid,fit):
    im=cv2.resize(image(fid),(640,360));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float)
    color=float(np.mean(np.std(im.astype(float),axis=2)));bank=[x for x in empty_bank(fit) if x[0]!=fid]
    ranked=sorted(bank,key=lambda x:abs(x[2]-gray.mean())+.8*abs(x[3]-color))[:3]
    if not ranked:return np.zeros((360,640),float),0.
    bg=np.median([x[1] for x in ranked],axis=0).astype(np.uint8)
    bggray=cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY).astype(float)
    # Compare relative brightness locally, plus chroma where both frames are color.
    a=gray-cv2.GaussianBlur(gray,(0,0),15);b=bggray-cv2.GaussianBlur(bggray,(0,0),15)
    diff=abs(a-b)
    if color>5 and np.mean([x[3] for x in ranked])>5:
        lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(float);bl=cv2.cvtColor(bg,cv2.COLOR_BGR2LAB).astype(float)
        diff+=.6*np.linalg.norm(lab[:,:,1:]-bl[:,:,1:],axis=2)
    mask=(diff>22).astype(np.uint8);mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    return mask.astype(float),float(len(ranked))

def relations(r):
    b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2;area=wh.prod()
    f=dict(log_width=np.log(wh[0]),log_height=np.log(wh[1]),log_area=np.log(area),aspect=wh[0]/wh[1],cx=cc[0]/1280,cy=cc[1]/720,
        edge=float(min(b[0],b[1],1280-b[2],720-b[3])<3))
    objects=frame_objects(r.frame_id)
    for cls,rr in objects.items():
        rb=rr[B].to_numpy();cent=(rb[:,:2]+rb[:,2:])/2
        if len(rb):
            inter=np.maximum(0,np.minimum(b[2:],rb[:,2:])-np.maximum(b[:2],rb[:,:2])).prod(1)
            f[cls+'_best_iou']=float(iou(b,rb).max());f[cls+'_coverage']=float((inter/area).max())
            f[cls+'_centers']=float(((cent>=b[:2])&(cent<=b[2:])).all(1).sum())
            f[cls+'_distance']=float(np.linalg.norm((cent-cc)/np.maximum(rb[:,2:]-rb[:,:2],10),axis=1).min())
        else:
            f.update({cls+'_best_iou':0.,cls+'_coverage':0.,cls+'_centers':0.,cls+'_distance':9.})
    tables=objects['table'][B].to_numpy()
    if len(tables):
        tc=(tables[:,:2]+tables[:,2:])/2;tw=tables[:,2:]-tables[:,:2]
        j=np.argmin(np.linalg.norm((tc-cc)/tw,axis=1));tb=tables[j];ts=tw[j]
        f.update(table_rel_x=(cc[0]-tc[j,0])/ts[0],table_rel_y=(cc[1]-tc[j,1])/ts[1],
            table_size_ratio=area/ts.prod(),table_bottom_gap=(b[3]-tb[3])/ts[1],table_top_gap=(b[1]-tb[1])/ts[1])
    hh=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy();hc=(hh[:,:2]+hh[:,2:])/2;hw=hh[:,2:]-hh[:,:2]
    inside=((hc>=b[:2])&(hc<=b[2:])).all(1);f['head_count']=int(inside.sum())
    if len(hh):
        j=np.argmin(np.linalg.norm((hc-cc)/np.maximum(hw,10),axis=1));f['nearest_head_distance']=float(np.linalg.norm((hc[j]-cc)/np.maximum(hw[j],10)))
        f['size_to_head']=float(area/hw[j].prod())
    wrist=[];body=[]
    for pr,kp in frame_pose(r.frame_id):
        for elbow,k in [(7,9),(8,10)]:
            if min(kp[k,2],kp[elbow,2])<.55:continue
            axis=kp[k,:2]-kp[elbow,:2];length=max(np.linalg.norm(axis),8);delta=cc-kp[k,:2]
            dist=np.linalg.norm(delta)/max(np.sqrt(area),15)
            if dist<3:
                projection=float(delta@axis/length**2);perp=float(abs(np.cross(axis,delta))/length**2)
                wrist.append((dist,projection,perp,kp[k,2],pr.view,kp[k,0],kp[k,1],length))
        valid=kp[:5,2]>.45
        if valid.sum()<2:continue
        pc=np.average(kp[:5,:2][valid],axis=0,weights=kp[:5,2][valid])
        if not ((pc>=b[:2])&(pc<=b[2:])).all():continue
        present=kp[:,2]>=.65;present[:5]=False
        if present.sum()<3:continue
        points=kp[present,:2];lo=np.minimum(b[:2],points.min(0));hi=np.maximum(b[2:],points.max(0))
        min_area=(hi-lo).prod();outside=((points<b[:2])|(points>b[2:])).any(1)
        body.append((area/min_area,float(outside.mean()),pr.view,float(np.maximum(points[:,1]-b[3],0).max()/wh[1]),int((kp[11:,2]>=.65).sum())))
    if wrist:
        wrist.sort();d0=wrist[0][0];near=[v for v in wrist if v[0]<min(1.3,d0+.5)] or [wrist[0]]
        f.update(wrist_distance=wrist[0][0],wrist_projection=float(np.median([v[1] for v in near])),
            wrist_perp=float(np.median([v[2] for v in near])),wrist_views=len(set(v[4] for v in near)),
            wrist_distal_fraction=float(np.mean([v[1]>-.15 for v in near])),wrist_conf=float(np.mean([v[3] for v in near])),
            hand_size_to_forearm=float(np.sqrt(area)/np.median([v[7] for v in near])))
    else:f.update(wrist_distance=9.,wrist_projection=-9.,wrist_perp=9.,wrist_views=0.,wrist_distal_fraction=0.,wrist_conf=0.,hand_size_to_forearm=0.)
    if body:
        # q75 is conservative: one rogue skeleton cannot create a truncation claim.
        f.update(body_point_iou_q75=float(np.quantile([v[0] for v in body],.75)),body_point_iou_min=min(v[0] for v in body),
            body_outside_fraction=float(np.median([v[1] for v in body])),body_pose_views=len(set(v[2] for v in body)),
            body_bottom_gap=float(np.median([v[3] for v in body])),body_leg_points=float(np.median([v[4] for v in body])))
    else:f.update(body_point_iou_q75=1.,body_point_iou_min=1.,body_outside_fraction=0.,body_pose_views=0.,body_bottom_gap=0.,body_leg_points=0.)
    return f

def pixels(r,fit):
    b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];area=wh.prod()
    lab,gray,p=pixel_maps(r.frame_id,fit);im=crop(image(r.frame_id),b);g=crop(gray,b);sk=crop(p,b)
    f=dict(color_strength=float(np.mean(np.std(im.astype(float),axis=2))),brightness=float(g.mean()/255),
        glare=float((g>240).mean()),contrast=float(g.std()/255),skin_mean=float(sk.mean()),skin_p90=float(np.quantile(sk,.9)))
    # Relative local chroma likelihood and component geometry; never a hard skin-color decision.
    cr=clipbox(np.r_[b[:2]-.5*wh,b[2:]+.5*wh]);sp=crop(p,cr)
    cut=max(.505,float(np.quantile(sp,.65)));mask=(sp>cut).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    n,labels,stats,cents=cv2.connectedComponentsWithStats(mask)
    best=None;bestscore=-1
    for j in range(1,n):
        x,y,w,hh,aa=stats[j];bb=np.array([x,y,x+w,y+hh])+np.tile(cr[:2],2)
        ov=float(iou(b,bb)[0,0]);score=ov*min(1,aa/max(area*.15,1))
        if score>bestscore:bestscore=score;best=(bb,aa,w,hh)
    if best:
        bb,aa,w,hh=best;f.update(skin_component_iou=float(iou(b,bb)[0,0]),skin_component_area=aa/area,
            skin_component_solidity=aa/(w*hh),skin_component_aspect=w/hh)
    else:f.update(skin_component_iou=0.,skin_component_area=0.,skin_component_solidity=0.,skin_component_aspect=0.)
    # Contours catch round plates, rectangular support and crop boundary crossings.
    resized=cv2.resize(g,(96,96));edges=cv2.Canny(resized,60,140);contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    circ=[];rect=[];sol=[]
    for co in contours:
        ar=cv2.contourArea(co);per=cv2.arcLength(co,True)
        if ar<96*96*.05 or per<1:continue
        (_, _),(rw,rh),_=cv2.minAreaRect(co)
        circ.append(4*np.pi*ar/per**2);rect.append(ar/max(rw*rh,1));sol.append(ar/max(cv2.contourArea(cv2.convexHull(co)),1))
    f.update(edge_density=float((edges>0).mean()),contour_round=max(circ,default=0.),contour_rect=max(rect,default=0.),contour_solid=max(sol,default=0.))
    fg,nref=foreground(r.frame_id,fit);f['fg_references']=nref;f['fg_inside']=float(crop(fg,b/2).mean())
    for side,bb in [('bottom',[b[0],b[3],b[2],min(720,b[3]+.5*wh[1])]),
                    ('top',[b[0],max(0,b[1]-.25*wh[1]),b[2],b[1]]),
                    ('left',[max(0,b[0]-.25*wh[0]),b[1],b[0],b[3]]),
                    ('right',[b[2],b[1],min(1280,b[2]+.25*wh[0]),b[3]])]:
        bb=np.asarray(bb);f['fg_'+side]=float(crop(fg,bb/2).mean()) if (bb[2:]>bb[:2]).all() else 0.
    return f

@lru_cache(None)
def body_bank(fit):
    out=[]
    for r in h[h.day.eq(fit)&h.class_name.eq('person')].itertuples():
        b=np.array([getattr(r,c) for c in B]);hh=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy();cc=(hh[:,:2]+hh[:,2:])/2
        inside=((cc>=b[:2])&(cc<=b[2:])).all(1)
        if inside.sum()!=1:continue
        hb=hh[inside][0];hc=(hb[:2]+hb[2:])/2;hw=hb[2:]-hb[:2]
        tables=frame_objects(r.frame_id)['table'][B].to_numpy()
        if not len(tables):continue
        tc=(tables[:,:2]+tables[:,2:])/2;tw=tables[:,2:]-tables[:,:2]
        j=np.argmin(np.linalg.norm((tc-hc)/tw,axis=1));rel=(hc-tc[j])/tw[j]
        out.append(dict(fid=r.frame_id,box=b,hc=hc,hw=hw,table=tables[j],rel=rel))
    return out

def template(r,fit):
    f=dict(seat_support=0.,seat_iou_max=1.,seat_iou_q75=1.,seat_bottom_min_gap=0.,seat_height_min_ratio=1.)
    if r.class_name!='person':return f
    b=np.array([getattr(r,c) for c in B]);hh=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy();cc=(hh[:,:2]+hh[:,2:])/2
    inside=((cc>=b[:2])&(cc<=b[2:])).all(1)
    if inside.sum()!=1:return f
    hb=hh[inside][0];hc=(hb[:2]+hb[2:])/2;hw=hb[2:]-hb[:2]
    tables=frame_objects(r.frame_id)['table'][B].to_numpy()
    if not len(tables):return f
    tc=(tables[:,:2]+tables[:,2:])/2;tw=tables[:,2:]-tables[:,:2]
    j=np.argmin(np.linalg.norm((tc-hc)/tw,axis=1));rel=(hc-tc[j])/tw[j]
    candidates=[]
    for a in body_bank(fit):
        if a['fid']==r.frame_id:continue
        dist=np.linalg.norm(rel-a['rel'])+.4*abs(np.log(hw[0]/a['hw'][0]))+abs(hc[0]-a['hc'][0])/500
        if dist>.65 or np.sign(rel[1])!=np.sign(a['rel'][1]):continue
        scale=np.clip(hw[0]/a['hw'][0],.7,1.4);bb=(a['box']-np.tile(a['hc'],2))*scale+np.tile(hc,2)
        candidates.append((dist,bb))
    candidates=sorted(candidates,key=lambda v:v[0])[:8]
    if len(candidates)<3:return f
    bb=np.array([v[1] for v in candidates]);ov=iou(b,bb)[0];heights=bb[:,3]-bb[:,1]
    f.update(seat_support=len(candidates),seat_iou_max=float(ov.max()),seat_iou_q75=float(np.quantile(ov,.75)),
        seat_bottom_min_gap=float(((bb[:,3]-b[3])/heights).min()),seat_height_min_ratio=float((b[3]-b[1])/heights.min()))
    return f

def features(rows,fit):
    output=[]
    for j,r in enumerate(rows.itertuples()):
        b=np.array([getattr(r,c) for c in B]);cc=(b[:2]+b[2:])/2
        bank=h[h.day.eq(fit)&h.class_name.eq(r.class_name)&h.frame_id.ne(r.frame_id)].copy();bank['prior_weight']=1.
        if r.class_name=='nude_hands':
            extra=pool[pool.day.eq(fit)&pool.frame_id.ne(r.frame_id)].copy();extra['prior_weight']=.35
            bank=pd.concat([bank,extra],ignore_index=True)
        if len(bank):
            bb=bank[B].to_numpy();center=(bb[:,:2]+bb[:,2:])/2;logsize=np.log(np.maximum(bb[:,2:]-bb[:,:2],1))
            w=np.exp(-.5*np.sum(((center-cc)/[220,150])**2,axis=1))*bank.prior_weight.to_numpy();w=w/w.sum()
            mean=w@logsize;std=np.sqrt(w@((logsize-mean)**2)+.08)
            delta=(np.log(b[2:]-b[:2])-mean)/std
        else:delta=np.zeros(2)
        out=dict(id=r.id,**relations(r),**pixels(r,fit),**template(r,fit),size_width_z=float(delta[0]),size_height_z=float(delta[1]));output.append(out)
        if j%750==0:print('FEATURES',fit,j,len(rows),flush=True)
    return pd.DataFrame(output).fillna(0.)

if __name__=='__main__':
    # V22 records contain official GT, realistic perturbations and real fit-day queries.
    # Pixel training on annotated screenshots would leak colored borders; exclude those.
    records=pd.read_csv(V22/'inputs/spatial_records.csv')
    for fit,held in zip(DAYS,DAYS[::-1]):
        tr=records[records.split.eq('train')&records.day.eq(fit)&records.frame_id.isin(paths)].copy()
        assert not set(tr.frame_id)&set(te.frame_id) and set(tr.day)=={fit}
        # Keep all originals and near-boundary positives; cap correlated synthetic variants.
        fixed=tr[tr.kind.isin(['gt','real','real_query'])];rest=tr.drop(fixed.index)
        # Use stable record indices, retaining each record's class.
        selected=[]
        for _,g in tr.drop(fixed.index).groupby('class_name'):
            selected.extend(g.sample(min(len(g),650),random_state=2301).index)
        tr=pd.concat([fixed,tr.loc[selected]],ignore_index=True)
        tr.to_csv(A/f'inputs/train_{fit}.csv',index=False)
        for split,rows in [('train',tr),('held',q[q.day.eq(held)]),('test',te)]:
            path=A/f'features/{split}_{fit}.csv'
            if path.exists():continue
            features(rows,fit).to_csv(path,index=False)
            print('SAVED',path,flush=True)
    (A/'inputs/extra_annotations_usage.json').write_text(json.dumps(dict(
        reviewed_pool_hand_boxes=len(pool),used_in_baseline_yolo=True,
        raw_pool_images_locally_available=False,annotated_contact_sheets_excluded_from_pixel_training=True,
        pool_use_this_round='low-weight spatial size prior in CV/context models; no pixel training on annotated screenshots',
        manual_frames=len(h.frame_id.unique()),training_day_exclusion=True),indent=2))
