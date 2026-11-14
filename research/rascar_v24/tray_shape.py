"""Test a plate-like elliptical rim inside an incomplete tray box."""
from common24 import *
from features23 import image,frame_objects
import cv2
cv2.setNumThreads(2)

def describe(r):
    b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2]
    xx,yy=np.meshgrid(np.linspace(b[0],b[2],128).astype('float32'),np.linspace(b[1],b[3],128).astype('float32'))
    im=cv2.remap(image(r.frame_id),xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);gray=cv2.GaussianBlur(gray,(3,3),.7)
    ellipses=[];straight=[]
    for low,high in [(30,70),(60,130),(100,190)]:
        edge=cv2.Canny(gray,low,high);contours,_=cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        best=0.
        for contour in contours:
            if len(contour)<35:continue
            (cx,cy),(wa,ha),angle=cv2.fitEllipse(contour)
            if min(wa,ha)<20 or max(wa,ha)>170 or max(wa,ha)/min(wa,ha)>4 or not 20<cx<108 or not 20<cy<108:continue
            points=contour[:,0].astype(float)-[cx,cy];theta=np.deg2rad(angle);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            uv=points@rot/[wa/2,ha/2];rad=np.linalg.norm(uv,axis=1);near=abs(rad-1)<.12
            bins=np.floor((np.arctan2(uv[near,1],uv[near,0])+np.pi)/(2*np.pi)*16).astype(int)%16
            coverage=len(set(bins))/16;area=np.pi*wa*ha/4/(128*128)
            support=float(near.mean())*coverage
            if .15<area<.85:best=max(best,support)
        ellipses.append(best)
        lines=cv2.HoughLinesP(edge,1,np.pi/180,threshold=25,minLineLength=45,maxLineGap=7)
        lengths=[]
        if lines is not None:
            for x1,y1,x2,y2 in lines[:,0]:
                if min(x1,x2,y1,y2)<5 or max(x1,x2,y1,y2)>123:continue
                lengths.append(np.hypot(x2-x1,y2-y1)/128)
        straight.append(sum(sorted(lengths,reverse=True)[:4]))
    rr=frame_objects(r.frame_id)['tray'];large=0.;rat=1.
    for ref in rr.itertuples():
        bb=np.array([getattr(ref,c) for c in B]);inter=np.maximum(0,np.minimum(b[2:],bb[2:])-np.maximum(b[:2],bb[:2])).prod();ratio=(bb[2:]-bb[:2]).prod()/wh.prod()
        if inter/wh.prod()>.85 and ratio>1.5 and ref.confidence>.5:large=max(large,ref.confidence);rat=max(rat,ratio)
    return dict(id=r.id,ellipse=float(np.median(ellipses)),straight=float(np.median(straight)),larger_tray_conf=large,larger_tray_area_ratio=rat)

if __name__=='__main__':
    fs={}
    for split,rows in [('oof',q),('test',te)]:
        d=pd.DataFrame([describe(r) for r in rows[rows.class_name.eq('tray')].itertuples()]).set_index('id').reindex(rows.id).fillna(0).reset_index()
        d.to_csv(A/f'features/tray_shape_{split}.csv',index=False);fs[split]=d
    for rule in ['oval_no_lines','oval_nested']:
        for cut in [.65,.8]:
            for weight in [.25,.5]:
                outputs=[]
                for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
                    f=fs[split];gate=rows.class_name.eq('tray').to_numpy()&f.ellipse.ge(cut).to_numpy()
                    gate&=f.straight.lt(.7).to_numpy() if rule=='oval_no_lines' else f.larger_tray_conf.ge(.55).to_numpy()&f.larger_tray_area_ratio.ge(2).to_numpy()
                    outputs.append(base+weight*np.maximum(.9-base,0)*gate)
                register(f'tray_{rule}_cut{cut}_w{weight}',*outputs,dict(type='ellipse_and_line_geometry',rule=rule,ellipse_min=cut,weight=weight))
