"""Retrieve reference body extent using head-anchored appearance and current table.

The query body's size is not a retrieval feature. Only fit-day manual boxes
supply reference extents; test photographs never supply target coordinates.
"""
from common24 import *
from features23 import image,heads,frame_objects,frame_pose
from functools import lru_cache
from skimage.feature import hog
import cv2
cv2.setNumThreads(2)

@lru_cache(None)
def descriptor(fid,index):
    hh=heads[heads.frame_id.eq(fid)].reset_index(drop=True);hb=hh.iloc[index][B].to_numpy(float);hc=(hb[:2]+hb[2:])/2;hw=hb[2:]-hb[:2]
    width=max(hw[0],12);cr=np.r_[hc+[-3*width,-width],hc+[3*width,7*width]]
    xx,yy=np.meshgrid(np.linspace(cr[0],cr[2],72).astype('float32'),np.linspace(cr[1],cr[3],96).astype('float32'))
    im=cv2.remap(image(fid),xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    appearance=hog(gray,orientations=8,pixels_per_cell=(12,12),cells_per_block=(1,1));appearance/=max(np.linalg.norm(appearance),1e-8)
    tables=frame_objects(fid)['table'][B].to_numpy();rel=np.zeros(2);side=0
    if len(tables):
        tc=(tables[:,:2]+tables[:,2:])/2;tw=tables[:,2:]-tables[:,:2];j=np.linalg.norm((tc-hc)/tw,axis=1).argmin();rel=(hc-tc[j])/tw[j];side=-1 if rel[1]<-.15 else 1 if rel[1]>.15 else 0
    candidates=[]
    for pr,kp in frame_pose(fid):
        good=kp[:5,2]>.45
        if good.sum()<2:continue
        center=np.average(kp[:5,:2][good],axis=0,weights=kp[:5,2][good]);distance=np.linalg.norm((center-hc)/hw)
        if distance<1.4:candidates.append((distance,kp))
    coords=np.zeros((12,2));valid=np.zeros(12,bool)
    if candidates:
        kp=min(candidates,key=lambda v:v[0])[1];valid=kp[5:,2]>=.55;coords=(kp[5:,:2]-hc)/width
    return dict(hc=hc,hw=hw,appearance=appearance,rel=rel,side=side,pose=coords,valid=valid)

def anchor(fid,b):
    hh=heads[heads.frame_id.eq(fid)][B].to_numpy();cent=(hh[:,:2]+hh[:,2:])/2
    inside=((cent>=b[:2])&(cent<=b[2:])).all(1)
    return int(np.flatnonzero(inside)[0]) if inside.sum()==1 else None

@lru_cache(None)
def bank(fit):
    out=[]
    for r in h[h.day.eq(fit)&h.class_name.eq('person')].itertuples():
        bb=np.array([getattr(r,c) for c in B]);j=anchor(r.frame_id,bb)
        if j is None:continue
        d=descriptor(r.frame_id,j);out.append(dict(fid=r.frame_id,box=bb,**d))
    return out

def features(rows,fit):
    records=[]
    for r in rows.itertuples():
        b=np.array([getattr(r,c) for c in B]);j=anchor(r.frame_id,b) if r.class_name=='person' else None
        f=dict(id=r.id,support=0,best_distance=9.,appearance_similarity=0.,iou_nearest=0.,iou_mean=0.,iou_max=0.,spread=1.,compatible_fraction=0.)
        for c in B:f['ref_'+c]=0.
        if j is not None:
            d=descriptor(r.frame_id,j);options=[]
            for other in bank(fit):
                if other['fid']==r.frame_id or other['side']!=d['side']:continue
                rel_dist=np.linalg.norm(other['rel']-d['rel']);spatial=np.linalg.norm((other['hc']-d['hc'])/[250,150]);scale=abs(np.log(d['hw'][0]/other['hw'][0]))
                if rel_dist>1 or spatial>2 or scale>.5:continue
                sim=float(other['appearance']@d['appearance']);valid=other['valid']&d['valid']
                pdist=float(np.minimum(np.linalg.norm(other['pose'][valid]-d['pose'][valid],axis=1),6).mean()/3) if valid.sum()>=3 else .7
                distance=1.6*(1-sim)+.3*rel_dist+.15*spatial+.25*scale+.25*pdist
                ref=(other['box']-np.tile(other['hc'],2))*(d['hw'][0]/other['hw'][0])+np.tile(d['hc'],2)
                options.append((distance,other['fid'],ref,sim))
            # Distinct frame support; do not count two people from one photo twice.
            chosen=[];seen=set()
            for option in sorted(options,key=lambda a:a[0]):
                if option[1] in seen:continue
                if option[0]>.95:continue
                chosen.append(option);seen.add(option[1])
                if len(chosen)==3:break
            if len(chosen)>=2:
                boxes=np.array([x[2] for x in chosen]);ov=iou(b,boxes)[0];weights=np.exp(-np.array([x[0] for x in chosen])*4);weights/=weights.sum()
                f.update(support=len(chosen),best_distance=chosen[0][0],appearance_similarity=chosen[0][3],iou_nearest=ov[0],iou_mean=float(weights@ov),iou_max=float(ov.max()),
                    compatible_fraction=float(weights@(ov>=.6)),spread=float(np.ptp(ov)),**dict(zip(['ref_'+v for v in B],chosen[0][2])))
        records.append(f)
    return pd.DataFrame(records)

if __name__=='__main__':
    os=[];ts=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        o=features(q[q.day.eq(held)],fit);t=features(te,fit);os.append(o);ts.append(t.set_index('id').loc[te.id])
        o.to_csv(A/f'features/seat_held_{fit}.csv',index=False);t.to_csv(A/f'features/seat_test_{fit}.csv',index=False)
        print('SEAT',fit,'templates',len(bank(fit)),'held support',o.support.ge(2).sum(),flush=True)
    fo=pd.concat(os).set_index('id').loc[q.id];ft=(ts[0]+ts[1])/2
    for reference in ['nearest','mean','max']:
        for direction in ['rescue','penalty','both']:
            for weight in [.25,.5]:
                outputs=[]
                for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
                    overlap=f['iou_'+reference].to_numpy();target=np.clip((.8-overlap)/.55,.05,.95)
                    eligible=rows.class_name.eq('person').to_numpy()&f.support.ge(2).to_numpy()&f.best_distance.le(.75).to_numpy()
                    lo=(overlap>=.65)&(f.iou_max.to_numpy()>=.7)&(f.spread.to_numpy()<.2)
                    hi=(overlap<.45)&(f.iou_max.to_numpy()<.55)&(f.spread.to_numpy()<.2)
                    if split=='test':
                        eligible&=np.minimum(ts[0].support,ts[1].support).to_numpy()>=2
                        lo&=np.minimum(ts[0]['iou_'+reference],ts[1]['iou_'+reference]).to_numpy()>=.6
                        hi&=np.maximum(ts[0].iou_max,ts[1].iou_max).to_numpy()<.55
                    gate=eligible&(lo if direction=='rescue' else hi if direction=='penalty' else lo|hi)
                    delta=np.minimum(target-base,0) if direction=='rescue' else np.maximum(target-base,0) if direction=='penalty' else target-base
                    outputs.append(base+weight*delta*gate)
                register(f'appearance_seat_{reference}_{direction}_w{weight}',*outputs,dict(type='head_anchored_appearance_retrieval',reference=reference,direction=direction,weight=weight))
