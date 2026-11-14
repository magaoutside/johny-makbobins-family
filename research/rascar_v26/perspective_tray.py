"""Position-dependent tray size and edge-supported completion.

Each fit day has independently annotated tabletop quadrilaterals from an empty
training frame. Local homographies provide image displacement for a rectangle
of fixed physical size at (x,y), including orientation and local lens effects.
No test labels, IDs, or query coordinates are fitted into a completion.
"""
from common26 import *
from features23 import image, frame_objects
from functools import lru_cache
from scipy.optimize import differential_evolution
import cv2
cv2.setNumThreads(2)

# Visually annotated TL, TR, BR, BL, on original fit-day images only.
# Units are relative to a regular square tabletop, not claimed centimetres.
# The narrow end table and double seat table have uncertain physical widths;
# both a four-regular-table and a six-table calibration are evaluated.
CALIBRATION = {
 '2026-10-25': dict(frame='GENERAL_mishka_1_5_20261025_000000', quads=[
  [[190,270],[244,261],[207,339],[149,337]],
  [[317,207],[494,200],[467,287],[281,290]],
  [[514,194],[625,192],[602,282],[487,281]],
  [[727,188],[850,191],[848,286],[714,284]],
  [[913,187],[1030,190],[1052,284],[925,284]],
  [[1106,194],[1211,203],[1238,292],[1146,292]]]),
 '2026-10-26': dict(frame='GENERAL_mishka_1_5_20261026_001200', quads=[
  [[193,233],[244,226],[209,296],[153,298]],
  [[312,207],[486,201],[456,286],[278,291]],
  [[518,195],[629,190],[619,279],[496,284]],
  [[718,187],[840,191],[839,286],[704,285]],
  [[921,186],[1038,191],[1058,288],[935,288]],
  [[1107,195],[1213,203],[1235,289],[1144,293]]])
}

def transform(points, H):
    points = np.asarray(points)
    p = np.concatenate([points, np.ones(points.shape[:-1]+(1,))], axis=-1) @ H.T
    return p[...,:2] / p[...,2:]

class CameraMetric:
    def __init__(self, day, mode='four'):
        self.day, self.mode = day, mode
        quads = np.asarray(CALIBRATION[day]['quads'], np.float32)
        ids = range(2,6) if mode == 'four' else range(6)
        self.maps=[]; self.centres=[]
        for i in ids:
            # In the physical mode the leftmost single table stays one unit
            # wide: its narrow image is foreshortening, not a narrow object.
            width = ([1,2,1,1,1,1] if mode in ['physical','monotone'] else [.5,1.5,1,1,1,1])[i]
            src = np.array([[0,0],[width,0],[width,1],[0,1]], np.float32)
            H = cv2.getPerspectiveTransform(src,quads[i])
            self.maps.append((H,np.linalg.inv(H)));self.centres.append(quads[i].mean(0))
        self.centres=np.asarray(self.centres)
        self.depth_rates=np.zeros(2)
        if mode=='monotone':
            # The rightmost far corner is partly hidden by a toy. Do not let
            # one uncertain quadrilateral reverse the depth trend. Estimate
            # it robustly from the three unobstructed regular centre tables.
            rates=[]
            for j in [2,3,4]:
                H,Hi=self.maps[j];sizes=[]
                for yy in [230.,290.]:
                    center=np.array([self.centres[j,0],yy]);world=transform(center,Hi)
                    pp=transform(world+np.array([[-.25,-.15],[.25,-.15],[.25,.15],[-.25,.15]]),H)
                    sizes.append(pp.max(0)-pp.min(0))
                rates.append(np.log(sizes[1]/sizes[0])/60.)
            self.depth_rates=np.maximum(0,np.median(rates,axis=0))

    def rectangle(self, center, length, short, angles):
        """Project a physical rectangle around an arbitrary image point.

        Blend locally centred displacements, never unrelated world origins.
        This models smooth changes of scale in both x and y. Extrapolation far
        outside the tabletop band is explicitly not used for score changes.
        """
        center=np.asarray(center,float); a=np.asarray(angles,float)
        destination=center.copy()
        if self.mode=='monotone':center=np.array([center[0],260.])
        corner=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])*[length,short]
        R=np.stack([np.stack([np.cos(a),-np.sin(a)],-1),np.stack([np.sin(a),np.cos(a)],-1)],-2)
        offset=np.einsum('...ij,kj->...ki',R,corner)
        d=(self.centres-center)/[150,150]
        logw=-np.sum(d*d,axis=1)/2
        weights=np.exp(logw-logw.max());weights/=weights.sum()
        out=np.zeros_like(offset)
        for weight,(H,Hi) in zip(weights,self.maps):
            world=transform(center,Hi)
            out+=weight*(transform(world+offset,H)-center)
        if self.mode=='monotone':
            # Smooth lateral distortion at reference depth; positive inverse
            # depth scales for width/height. All orientations grow toward the
            # lower image at fixed x, within the supported tabletop band.
            scale=1/np.maximum(.4,1-self.depth_rates*(destination[1]-260.))
            return out*scale+destination
        return out+center

def bounds(polys):
    return np.concatenate([polys.min(-2),polys.max(-2)],-1)

@lru_cache(None)
def fit_size(day, mode):
    cam=CameraMetric(day,mode)
    data=h[h.day.eq(day)&h.class_name.eq('tray')]
    angles=np.deg2rad(np.arange(0,180,3))
    bb=data[B].to_numpy();cent=(bb[:,:2]+bb[:,2:])/2;wh=bb[:,2:]-bb[:,:2]
    def objective(params):
        length,ratio=params;short=length/ratio
        errors=[]
        for c,size in zip(cent,wh):
            b=bounds(cam.rectangle(c,length,short,angles));p=b[:,2:]-b[:,:2]
            residual=np.log(np.maximum(p,1)/size)
            errors.append(np.min(np.mean(residual**2,axis=1)))
        return np.mean(errors)+.002*(ratio-1.6)**2
    result=differential_evolution(objective,[(.25,.95),(1.2,2.2)],seed=26,popsize=8,maxiter=45,tol=1e-6,polish=True)
    length,ratio=result.x
    out=dict(fit_day=day,mode=mode,length=float(length),short=float(length/ratio),aspect=float(ratio),fit_n=len(data),loss=float(result.fun),calibration_frame=CALIBRATION[day]['frame'])
    (A/f'models/tray_size_{mode}_{day}.json').write_text(json.dumps(out,indent=2))
    return out

@lru_cache(maxsize=80)
def edge_map(fid):
    im=image(fid);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    edge=cv2.Canny(gray,55,130)
    dist=cv2.distanceTransform(255-edge,cv2.DIST_L2,3)
    return np.exp(-.5*(dist/1.6)**2)

@lru_cache(maxsize=80)
def oriented_edge_map(fid):
    gray=cv2.cvtColor(image(fid),cv2.COLOR_BGR2GRAY)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    mag=np.sqrt(gx*gx+gy*gy)+1e-6
    edge=cv2.Canny(gray,45,110).astype(np.float32)/255
    return edge,gx/mag,gy/mag

def support(polys, field):
    ps=np.asarray(polys).reshape(-1,4,2)
    t=np.linspace(.08,.92,32)
    points=ps[:,:,None,:]*(1-t[None,None,:,None])+np.roll(ps,-1,axis=1)[:,:,None,:]*t[None,None,:,None]
    if isinstance(field,tuple):
        # A wood grain or a curved plate edge near a proposed side is not a
        # rectangular tray boundary. Require aligned gradients along the side.
        vectors=np.roll(ps,-1,axis=1)-ps
        normals=np.stack([-vectors[...,1],vectors[...,0]],-1)
        normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-6)
        offsets=np.arange(-2,3)
        pp=points[:,:,:,None,:]+normals[:,:,None,None,:]*offsets[None,None,None,:,None]
        ix=np.rint(pp).astype(int);xx=np.clip(ix[...,0],0,1279);yy=np.clip(ix[...,1],0,719)
        ed,gx,gy=field
        alignment=np.abs(gx[yy,xx]*normals[:,:,None,None,0]+gy[yy,xx]*normals[:,:,None,None,1])**4
        scores=ed[yy,xx]*alignment*np.exp(-.5*(offsets/1.7)**2)
        values=scores.max(-1).mean(-1)
        return values,np.sort(values,axis=1)[:,1]
    ix=np.rint(points).astype(int);x=np.clip(ix[...,0],0,1279);y=np.clip(ix[...,1],0,719)
    edge=field[y,x].mean(-1)
    return edge, np.sort(edge,axis=1)[:,1]

def rectangular_poly(b):
    x1,y1,x2,y2=np.asarray(b);return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])

def complete(fid, refbox, day, mode='four', policy='default'):
    refbox=np.asarray(refbox,float);center=(refbox[:2]+refbox[2:])/2;wh=refbox[2:]-refbox[:2]
    model=fit_size(day,mode);cam=CameraMetric(day,mode)
    # 36 rotations, three modest size variations, nine alignment hypotheses.
    polys=[]
    for scale in ([.75,.9,1.,1.1,1.25] if policy=='checked' or policy.startswith('oriented') else [.9,1.,1.1]):
        pp=cam.rectangle(center,model['length']*scale,model['short']*scale,np.deg2rad(np.arange(0,180,5)))
        bb=bounds(pp);newwh=bb[:,2:]-bb[:,:2]
        for ax in [-.5,0,.5]:
            for ay in [-.5,0,.5]:
                shift=(newwh-wh)*[ax,ay]
                polys.append(pp+shift[:,None,:])
    polys=np.concatenate(polys);bb=bounds(polys);ar=(bb[:,2:]-bb[:,:2]).prod(1);area=wh.prod()
    intersection=np.maximum(0,np.minimum(bb[:,2:],refbox[2:])-np.maximum(bb[:,:2],refbox[:2])).prod(1)
    coverage=intersection/area;sizechange=ar/area
    field=oriented_edge_map(fid) if policy.startswith('oriented') else edge_map(fid)
    edges,third=support(polys,field);oldedges,oldthird=support(rectangular_poly(refbox),field)
    if policy=='checked' or policy.startswith('oriented'):
        # Fair competing hypothesis: an intact tray can be rotated. Comparing
        # its axis-aligned bounding box edges with a rotated extension unfairly
        # favours expansion. Fit rotated polygons INSIDE the reference bounds.
        op=cam.rectangle(center,model['length'],model['short'],np.deg2rad(np.arange(0,180,5)))
        ob=bounds(op);owh=ob[:,2:]-ob[:,:2];stretch=wh/owh
        op=(op-(ob[:,:2]+ob[:,2:])[:,None,:]/2)*stretch[:,None,:]+center
        oe,_=support(op,field)
        # Modest independent width/height tolerance for imperfect calibration.
        plausible=(stretch.max(1)/stretch.min(1)<1.55)&(stretch.min(1)>.55)&(stretch.max(1)<1.6)
        oldscore=.6*oe.min(1)+.4*oe.mean(1);oldscore[~plausible]=0
        oldthird=np.array([oldscore.max()])
        third=.6*edges.min(1)+.4*edges.mean(1)
    # Three supported edges, containment, small movement, and limited expansion.
    center_shift=np.linalg.norm(((bb[:,:2]+bb[:,2:])/2-center)/wh,axis=1)
    valid=(coverage>=.80)&(sizechange>=.92)&(sizechange<=2.6)&(bb[:,0]>=0)&(bb[:,1]>=0)&(bb[:,2]<=1280)&(bb[:,3]<=720)
    if policy=='oriented_grouped':
        # Extending one tray must not absorb another tray. Only independent
        # detector references are used here, never held-day manual boxes.
        neighbors=frame_objects(fid)['tray']
        neighbors=neighbors[neighbors.confidence.ge(.5)]
        if len(neighbors):
            rb=neighbors[B].to_numpy();centers=(rb[:,:2]+rb[:,2:])/2
            same_object=iou(refbox,rb)[0]>=.3
            outside=~((centers>=refbox[:2])&(centers<=refbox[2:])).all(1)
            centers=centers[~same_object&outside]
            if len(centers):
                absorbed=((centers[None,:,:]>=bb[:,None,:2])&(centers[None,:,:]<=bb[:,None,2:])).all(2).any(1)
                valid &= ~absorbed
    merit=.65*third+.35*edges.mean(1)-.08*center_shift-.08*abs(np.log(sizechange))-.15*(1-coverage)
    merit[~valid]=-1e6
    k=int(merit.argmax())
    if not valid[k]:
        return dict(box=refbox.tolist(),poly=rectangular_poly(refbox).tolist(),edge=0.,old_edge=float(oldthird[0]),area_ratio=1.,gain=0.,valid=False)
    return dict(box=bb[k].tolist(),poly=polys[k].tolist(),edge=float(third[k]),old_edge=float(oldthird[0]),area_ratio=float(sizechange[k]),gain=float(third[k]-oldthird[0]),valid=True)

def accept(c, center, cut=.35, gain=.08):
    return c['valid'] and c['edge']>=cut and c['gain']>=gain and c['area_ratio']>=1.12 and 140<=center[0]<=1260 and 190<=center[1]<=400

def benchmark(mode,policy='default'):
    records=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        model=fit_size(fit,mode)
        for index,r in h[h.day.eq(held)&h.class_name.eq('tray')].iterrows():
            gt=r[B].to_numpy(float);wh=gt[2:]-gt[:2]
            cases=[('intact',0.,gt.copy())]
            for frac in [.2,.35,.5]:
                for side in range(4):
                    partial=gt.copy();partial[side]+=(1 if side<2 else -1)*frac*wh[side%2]
                    cases.append((str(side),frac,partial))
            for side,frac,partial in cases:
                c=complete(r.frame_id,partial,fit,mode,policy);gated=accept(c,(partial[:2]+partial[2:])/2)
                box=np.array(c['box']) if gated else partial
                records.append(dict(frame_id=r.frame_id,gt_index=index,day=held,mode=mode,side=side,fraction=frac,
                    before=float(iou(gt,partial)[0,0]),raw_after=float(iou(gt,c['box'])[0,0]),after=float(iou(gt,box)[0,0]),changed=gated,
                    edge=c['edge'],gain=c['gain'],**dict(zip(B,box)),**{'old_'+b:partial[j] for j,b in enumerate(B)},**{'gt_'+b:gt[j] for j,b in enumerate(B)}))
    key=mode if policy=='default' else mode+'_'+policy
    d=pd.DataFrame(records);d.to_csv(A/f'results/tray_completion_benchmark_{key}.csv',index=False)
    print('BENCH',key,d.groupby(['day','fraction'])[['before','raw_after','after','changed']].mean().round(4).to_string(),flush=True)
    return d

def features(rows,fit,mode,policy='default'):
    records=[]
    for r in rows.itertuples():
        f=dict(id=r.id,eligible=0,edge=0.,gain=0.,area_ratio=1.,old_iou=0.,new_iou=0.,ref_x1=0.,ref_y1=0.,ref_x2=0.,ref_y2=0.)
        if r.class_name!='tray':records.append(f);continue
        b=np.array([getattr(r,k) for k in B]);rr=frame_objects(r.frame_id)['tray'];rr=rr[rr.confidence.ge(.5)]
        if not len(rr):records.append(f);continue
        rb=rr.iloc[int(iou(b,rr[B])[0].argmax())][B].to_numpy(float)
        c=complete(r.frame_id,rb,fit,mode,policy);center=(rb[:2]+rb[2:])/2
        f.update(eligible=int(accept(c,center,0.,-.5)),edge=c['edge'],gain=c['gain'],area_ratio=c['area_ratio'],
                 old_iou=float(iou(b,rb)[0,0]),new_iou=float(iou(b,c['box'])[0,0]),**dict(zip(['ref_'+k for k in B],c['box'])))
        records.append(f)
    return pd.DataFrame(records).set_index('id')

def run(modes=None,policy='default'):
    (A/'inputs/table_corner_calibration.json').write_text(json.dumps(CALIBRATION,indent=2))
    for mode in (modes or ['four','six','physical']):
        key=mode if policy=='default' else mode+'_'+policy
        benchmark(mode,policy);of=[];tf=[]
        for fit,held in zip(DAYS,DAYS[::-1]):
            o=features(q[q.day.eq(held)],fit,mode,policy);t=features(te,fit,mode,policy)
            o.to_csv(A/f'features/tray_{key}_held_{fit}.csv');t.to_csv(A/f'features/tray_{key}_test_{fit}.csv');of.append(o);tf.append(t)
        fo=pd.concat(of).loc[q.id];ft=(tf[0]+tf[1])/2
        for cut in [.3,.45]:
            for gain in [0.,.08]:
                for weight in [.2,.4]:
                    scores=[]
                    for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
                        gate=f.eligible.ge(.99).to_numpy()&f.edge.ge(cut).to_numpy()&f.gain.ge(gain).to_numpy()&guard(split,rows,'tray')
                        old=np.clip((.7-f.old_iou.to_numpy())/.4,0,1);new=np.clip((.7-f.new_iou.to_numpy())/.4,0,1)
                        scores.append(np.clip(base+weight*(new-old)*gate,0,1))
                    register(f'perspective_{key}_e{cut}_g{gain}_w{weight}',*scores,dict(type='local_projective_fixed_physical_size',mode=mode,policy=policy,edge=cut,gain=gain,weight=weight,fit='opposite-day manual labels and fit-day empty frame tabletop corners'))

if __name__=='__main__':
    args=sys.argv[1:];policy='oriented_grouped' if 'oriented_grouped' in args else 'oriented' if 'oriented' in args else 'checked' if 'checked' in args else 'default'
    run([x for x in args if x not in ['checked','oriented','oriented_grouped']] or None,policy)
