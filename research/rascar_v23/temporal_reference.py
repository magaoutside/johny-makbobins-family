"""Same-seat, similar-clothing reference propagation; detector outputs only.

Uses unlabeled frames of each inference day in a symmetric time window. No
manual box or visual audit label is used to make the propagated references.
"""
from common23 import *
from features23 import heads,frame_objects,image,crop
import cv2
from scipy.spatial.distance import cdist

def appearance(fid,hb):
    wh=hb[2:]-hb[:2];hc=(hb[:2]+hb[2:])/2
    bb=np.r_[hc+[-1.2*wh[0],.25*wh[1]],hc+[1.2*wh[0],2.3*wh[1]]]
    im=crop(image(fid),bb);im=cv2.resize(im,(32,32));lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)
    hist=cv2.calcHist([lab],[0,1,2],None,[4,8,8],[0,256,0,256,0,256]).ravel();hist=np.sqrt(hist/max(hist.sum(),1))
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float);gray=cv2.resize(gray,(8,8)).ravel();gray-=gray.mean();gray/=max(np.linalg.norm(gray),1)
    return hist,gray

records=[]
for fid,g in heads.groupby('frame_id'):
    hb=g[B].to_numpy();cent=(hb[:,:2]+hb[:,2:])/2
    rr=frame_objects(fid)['person'];stamp=pd.Timestamp(fid.rsplit('_',2)[-2]+fid.rsplit('_',1)[-1]) if False else pd.to_datetime(fid[-15:],format='%Y%m%d_%H%M%S')
    for j,b in enumerate(hb):
        hh,gray=appearance(fid,b);c=cent[j];providers=[]
        for r in rr.itertuples():
            pb=np.array([getattr(r,k) for k in B]);inside=((cent>=pb[:2])&(cent<=pb[2:])).all(1)
            if r.confidence>=.65 and inside.sum()==1 and inside[j]:providers.append((float(r.confidence),pb))
        provider=max(providers,key=lambda v:v[0])[1] if providers else None
        records.append(dict(fid=fid,day=fid[-15:-7],time=stamp.timestamp(),head=b,center=c,hist=hh,gray=gray,provider=provider))

def features(rows,minutes,similarity):
    result=[]
    for r in rows.itertuples():
        rec=dict(id=r.id,eligible=False,neighbors=0,overlap=1.,height_ratio=1.,top_gap=0.,appearance=0.,ref_x1=0.,ref_y1=0.,ref_x2=0.,ref_y2=0.,neighbor_frames='')
        if r.class_name!='person':result.append(rec);continue
        b=np.array([getattr(r,c) for c in B]);cands=[x for x in records if x['fid']==r.frame_id and ((x['center']>=b[:2])&(x['center']<=b[2:])).all()]
        if len(cands)!=1:result.append(rec);continue
        x=cands[0];neighbors=[]
        for v in records:
            if v['fid']==x['fid'] or v['day']!=x['day'] or v['provider'] is None or abs(v['time']-x['time'])>minutes*60:continue
            size=x['head'][2:]-x['head'][:2];other=v['head'][2:]-v['head'][:2]
            distance=np.linalg.norm((v['center']-x['center'])/[65,55]);scale=size[0]/other[0]
            color=float(x['hist']@v['hist']);structure=float(x['gray']@v['gray'])
            if distance>1 or not .7<scale<1.4 or color<similarity or structure<.1:continue
            bb=(v['provider']-np.tile(v['center'],2))*scale+np.tile(x['center'],2)
            neighbors.append((color,bb,v['fid']))
        if not neighbors:result.append(rec);continue
        # One box per neighboring frame; two views do not count as two frames.
        unique={}
        for v in sorted(neighbors,key=lambda x:x[0],reverse=True):unique.setdefault(v[2],v)
        neighbors=list(unique.values());bb=np.median([v[1] for v in neighbors],axis=0);wh=bb[2:]-bb[:2]
        rec.update(eligible=True,neighbors=len(neighbors),overlap=float(iou(b,bb)[0,0]),height_ratio=float((b[3]-b[1])/wh[1]),top_gap=float(abs(b[1]-bb[1])/wh[1]),
            appearance=float(np.median([v[0] for v in neighbors])),**dict(zip(['ref_'+c for c in B],bb)),neighbor_frames=';'.join(v[2] for v in neighbors))
        result.append(rec)
    return pd.DataFrame(result)

if __name__=='__main__':
    pd.DataFrame([dict(frame_id=x['fid'],day=x['day'],head_x=x['center'][0],head_y=x['center'][1],provider=x['provider'] is not None) for x in records]).to_csv(A/'inputs/temporal_reference_nodes.csv',index=False)
    for minutes in [15,30]:
        for similarity in [.7,.85]:
            fs={split:features(rows,minutes,similarity) for split,rows in [('oof',q),('test',te)]}
            for split,f in fs.items():f.to_csv(A/f'features/temporal_{minutes}_{similarity}_{split}.csv',index=False)
            for support in [1,2]:
                output=[]
                for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
                    f=fs[split];gate=f.eligible.to_numpy(bool)&f.neighbors.ge(support).to_numpy()&f.overlap.lt(.6).to_numpy()&f.height_ratio.lt(.8).to_numpy()&f.top_gap.lt(.2).to_numpy()
                    strength=np.clip((.7-f.overlap.to_numpy())/.3,0,1);s=base+.3*(1-base)*strength*gate;output.append(s)
                register(f'temporal_{minutes}_sim{similarity}_support{support}',*output,dict(type='same_seat_clothing_temporal',minutes=minutes,min_histogram_similarity=similarity,min_frames=support,
                    use_of_labels='none; cached pretrained detector outputs on unlabeled frames of the same inference day',validation_caveat='sparse annotated-day frame sampling differs from test; also test context is used at inference'))
