"""Independent local boundary proposals from opposite-day annotated exemplars.

No test annotations, scores, IDs, or query-box coordinates are used to generate
the proposals. Frozen SAM boxes only define the search region and scale range.
Templates carry the annotation boundary through normalized pixel matching.
"""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import cv2
ROOT=Path(__file__).resolve().parent.parent; A=ROOT/'rascar_v30'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import h,q,te,B,DAYS,iou
from features23 import image, crop, nms
cv2.setNumThreads(2)

def channels(im):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    gray=cv2.GaussianBlur(gray,(3,3),.6)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    return gray,np.hypot(gx,gy)

def bank(day,cls):
    out=[]
    for ix,r in h[h.day.eq(day)&h.class_name.eq(cls)].iterrows():
        b=r[B].to_numpy(float); wh=b[2:]-b[:2]
        # Fixed 15% context prevents the crop edge from becoming a fake edge.
        bb=np.r_[b[:2]-.15*wh,b[2:]+.15*wh]
        xx,yy=np.meshgrid(np.linspace(bb[0],bb[2],max(10,round(1.3*wh[0]))),
                         np.linspace(bb[1],bb[3],max(10,round(1.3*wh[1]))))
        patch=cv2.remap(image(r.frame_id),xx.astype('float32'),yy.astype('float32'),cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
        out.append(dict(frame_id=r.frame_id,annotation_index=int(ix),wh=wh,patch=patch))
    return out

def predict(fid,b,templates,cls):
    wh=b[2:]-b[:2]; pad=np.maximum(wh*.85,10)
    lo=np.maximum(0,np.floor(b[:2]-pad)).astype(int)
    hi=np.minimum([1280,720],np.ceil(b[2:]+pad)).astype(int)
    roi=image(fid)[lo[1]:hi[1],lo[0]:hi[0]]
    gray,mag=channels(roi); allmatches=[]
    for t in templates:
        if t['frame_id']==fid:continue
        tw=t['wh']; ratio=tw[0]/tw[1]
        if cls=='cup' and not .45<ratio<2.1:continue
        # Isotropic scaling preserves exemplar shape; never squeeze to query.
        nominal=np.sqrt(np.prod(wh)/np.prod(tw))
        best=None
        for factor in [.65,.8,1.,1.2,1.45]:
            scale=nominal*factor
            sz=np.maximum(8,np.round(tw*scale*1.3)).astype(int)
            if np.any(sz>=np.array([roi.shape[1],roi.shape[0]])):continue
            p=cv2.resize(t['patch'],tuple(sz),interpolation=cv2.INTER_LINEAR)
            pg,pm=channels(p)
            if pg.std()<.025:continue
            sim=.6*cv2.matchTemplate(gray,pg,cv2.TM_CCOEFF_NORMED)+.4*cv2.matchTemplate(mag,pm,cv2.TM_CCOEFF_NORMED)
            yy,xx=np.indices(sim.shape)
            cx=lo[0]+xx+sz[0]/2;cy=lo[1]+yy+sz[1]/2
            allowed=(abs(cx-(b[0]+b[2])/2)<.55*wh[0])&(abs(cy-(b[1]+b[3])/2)<.55*wh[1])
            sim[~allowed]=-1
            _,score,_,pos=cv2.minMaxLoc(sim)
            if best is None or score>best['similarity']:
                origin=lo+pos; outbox=np.r_[origin+sz*(.15/1.3),origin+sz*(1.15/1.3)]
                best=dict(source_frame=t['frame_id'],source_annotation=t['annotation_index'],similarity=float(score),
                          **dict(zip(B,outbox)))
        if best is not None:allmatches.append(best)
    if not allmatches:return []
    # Repeated crops of the same photograph cannot count as independent votes.
    m=pd.DataFrame(allmatches).sort_values('similarity',ascending=False).drop_duplicates('source_frame').head(5)
    return m.to_dict('records')

def run():
    (A/'inputs/exemplar_spec.json').write_text(json.dumps(dict(
        classes=['cup','nude_hands'],training='manual annotations of fit day only',
        template_context=.15,scales=[.65,.8,1,1.2,1.45],gray_weight=.6,gradient_weight=.4,
        search_anchor='frozen SAM, confidence>=0.45, NMS 0.5',test_labels_used=False),indent=2))
    rr={s:pd.read_csv(ROOT/f'rascar_v15/v8/fused_{s}_references.csv') for s in ['train','test']}
    for fit in DAYS:
        for split,rows in [('oof',q[q.day.ne(fit)]),('test',te)]:
            path=A/f'features/exemplars_{split}_fit{fit}.csv'
            if path.exists():continue
            raw=rr['train' if split=='oof' else 'test'];out=[];num=0
            for cls in ['cup','nude_hands']:
                templates=bank(fit,cls)
                for fid in sorted(rows[rows.class_name.eq(cls)].frame_id.unique()):
                    refs=nms(raw[raw.frame_id.eq(fid)&raw.class_name.eq(cls)&raw.confidence.ge(.45)],.5)
                    for ri,r in refs.iterrows():
                        for rank,p in enumerate(predict(fid,r[B].to_numpy(float),templates,cls)):
                            out.append(dict(fit_day=fit,frame_id=fid,class_name=cls,reference_index=int(ri),
                                ref_confidence=float(r.confidence),rank=rank,**{f'ref_{c}':r[c] for c in B},**p))
                        num+=1
                    if num%20<3:print('EXEMPLAR',fit,split,cls,num,flush=True)
            pd.DataFrame(out).to_csv(path,index=False)
            print('SAVED',path.name,len(out),flush=True)

if __name__=='__main__':run()
