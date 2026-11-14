"""Query-independent GrabCut and detector consensus, with joint assignment."""
from common24 import *
from features23 import image,frame_objects
from scipy.optimize import linear_sum_assignment
import cv2
cv2.setNumThreads(2)

def segment(fid,b):
    b=np.asarray(b,float);wh=b[2:]-b[:2];cr=np.r_[b[:2]-.5*wh,b[2:]+.5*wh]
    xx,yy=np.meshgrid(np.linspace(cr[0],cr[2],96).astype('float32'),np.linspace(cr[1],cr[3],96).astype('float32'))
    im=cv2.remap(image(fid),xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    outputs=[]
    for pad in [8,14]:
        mask=np.full((96,96),cv2.GC_PR_BGD,np.uint8);mask[:6]=0;mask[-6:]=0;mask[:,:6]=0;mask[:,-6:]=0
        mask[24:72,24:72]=cv2.GC_PR_FGD
        # Only the small object center is definite foreground; failures abstain.
        mask[48-pad//2:48+pad//2,48-pad//2:48+pad//2]=cv2.GC_FGD
        cv2.setRNGSeed(2401)
        cv2.grabCut(im,mask,None,np.zeros((1,65)),np.zeros((1,65)),3,cv2.GC_INIT_WITH_MASK)
        binary=((mask==1)|(mask==3)).astype(np.uint8)
        n,labels,stats,_=cv2.connectedComponentsWithStats(binary)
        # Keep the component through the center seed; it can be background.
        label=labels[48,48]
        if label==0:outputs.append((b,np.zeros((96,96),np.uint8),0.,1.));continue
        region=(labels==label).astype(np.uint8);ys,xs=np.nonzero(region)
        bb=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1])/96*np.tile(cr[2:]-cr[:2],2)+np.tile(cr[:2],2)
        touches=float(min(xs.min(),ys.min(),95-xs.max(),95-ys.max())<=7)
        outputs.append((bb,region,float(region.sum()/2304),touches))
    a,c=outputs;stability=float(iou(a[0],c[0])[0,0]);intersection=(a[1]&c[1]).sum();union=(a[1]|c[1]).sum()
    return dict(zip(['seg_'+v for v in B],(a[0]+c[0])/2))|dict(seg_stability=stability,mask_stability=float(intersection/max(union,1)),
        seg_area=(a[2]+c[2])/2,seg_touches=max(a[3],c[3]))

def references(split,rows):
    frames=set(rows.frame_id);refs=[]
    source_files=[('teacher',ROOT/f'rascar_v15/v8/fused_{"train" if split=="oof" else "test"}_references.csv'),
        ('yolo_expanded',ROOT/f'rascar_v20/hand_expanded_sahi_refs_{split}.csv'),
        ('yolo_frozen',ROOT/f'rascar_v22/zoom_frozen_refs_{split}.csv'),
        ('yolo_targeted',ROOT/f'rascar_v22/zoom_targeted_refs_{split}.csv'),
        ('yolo_objects',ROOT/f'rascar_v20/cup_tray_sahi_refs_{split}.csv')]
    for source,path in source_files:
        d=pd.read_csv(path);d=d[d.frame_id.isin(frames)&d.class_name.isin(CL)&d.confidence.ge(.45)].copy();d['source_name']=source
        refs.append(d[['frame_id','class_name','confidence','source_name']+B])
    refs=pd.concat(refs,ignore_index=True);out=[]
    for (fid,cls),g in refs.groupby(['frame_id','class_name']):
        clusters=[]
        for r in g.sort_values('confidence',ascending=False).itertuples():
            b=np.array([getattr(r,c) for c in B]);fits=[float(iou(b,np.median([x[0] for x in group],axis=0))[0,0]) for group in clusters]
            j=int(np.argmax(fits)) if fits else -1
            if fits and fits[j]>=.55:
                if r.source_name not in [x[2] for x in clusters[j]]:clusters[j].append((b,r.confidence,r.source_name))
            else:clusters.append([(b,r.confidence,r.source_name)])
        for j,cluster in enumerate(clusters):
            boxes=np.array([x[0] for x in cluster]);conf=np.array([x[1] for x in cluster]);sources=[x[2] for x in cluster]
            # The YOLO variants count as one correlated family.
            families=len(set('teacher' if v=='teacher' else 'yolo' for v in sources))
            teacher=np.array([s=='teacher' for s in sources]);weights=conf.copy()
            for family in [teacher,~teacher]:
                if family.any():weights[family]/=weights[family].sum()
            bb=np.average(boxes,weights=weights,axis=0)
            d=dict(frame_id=fid,class_name=cls,instance=j,confidence=float(conf.max()),families=families,
                sources='|'.join(sources),members=json.dumps(boxes.tolist()),**dict(zip(B,bb)))
            if 'teacher' in sources and conf[teacher].max()>=.55:
                d.update(segment(fid,boxes[np.flatnonzero(teacher)[0]]))
            else:d.update(dict(zip(['seg_'+v for v in B],bb)),seg_stability=0.,mask_stability=0.,seg_area=0.,seg_touches=1.)
            out.append(d)
    return pd.DataFrame(out)

def features(rows,refs):
    result=[]
    for (fid,cls),group in rows.groupby(['frame_id','class_name']):
        rr=refs[refs.frame_id.eq(fid)&refs.class_name.eq(cls)].reset_index(drop=True)
        ov=iou(group[B],rr[B]);assigned={}
        if len(rr):
            row,col=linear_sum_assignment(-ov)
            assigned=dict(zip(row,col))
        qq=group[B].to_numpy()
        for k,r in enumerate(group.itertuples()):
            f=dict(id=r.id,reference_iou=0.,consensus_min=0.,consensus_max=0.,consensus_mean=0.,families=0.,
                assigned=0.,rival_margin=0.,seg_iou=0.,seg_stability=0.,mask_stability=0.,seg_area=0.,seg_touches=1.,
                seg_cover=0.,ref_conf=0.,instances=0.,duplicate_loser=0.)
            for field in ['ref_','seg_']:
                for c in B:f[field+c]=0.
            if len(rr):
                j=int(ov[k].argmax());ref=rr.iloc[j];best=ov[k,j];rivals=np.delete(ov[:,j],k)
                members=np.array(json.loads(ref.members));ious=iou(qq[k],members)[0]
                center=(rr[['x1','y1']].to_numpy()+rr[['x2','y2']].to_numpy())/2
                count=((center>=qq[k,:2])&(center<=qq[k,2:])).all(1).sum()
                seg=ref[['seg_'+c for c in B]].to_numpy(float);inter=np.maximum(0,np.minimum(qq[k,2:],seg[2:])-np.maximum(qq[k,:2],seg[:2])).prod()
                f.update(reference_iou=float(best),consensus_min=float(ious.min()),consensus_max=float(ious.max()),consensus_mean=float(ious.mean()),
                    families=int(ref.families),assigned=int(assigned.get(k)==j),rival_margin=float(best-(rivals.max() if len(rivals) else 0)),
                    seg_iou=float(iou(qq[k],seg)[0,0]),seg_cover=float(inter/max((seg[2:]-seg[:2]).prod(),1)),
                    ref_conf=float(ref.confidence),instances=int(count),duplicate_loser=int(len(rivals)>0 and rivals.max()>best+.12 and best>.2),
                    **{v:float(ref[v]) for v in ['seg_stability','mask_stability','seg_area','seg_touches']},
                    **dict(zip(['ref_'+v for v in B],ref[B])),**dict(zip(['seg_'+v for v in B],seg)))
            result.append(f)
    return pd.DataFrame(result).set_index('id').loc[rows.id].reset_index()

if __name__=='__main__':
    import sys
    fs={}
    for split,rows in [('oof',q),('test',te)]:
        path=A/f'features/instances_{split}.csv'
        if path.exists() and '--rebuild' not in sys.argv:fs[split]=pd.read_csv(path);continue
        refs=references(split,rows);refs.to_csv(A/f'features/instance_references_{split}.csv',index=False)
        fs[split]=features(rows,refs);fs[split].to_csv(path,index=False);print('INSTANCES',split,len(refs),flush=True)
    for rule in ['consensus_rescue','consensus_penalty','group_loser','segmentation_rescue','segmentation_penalty']:
        for cls in CL:
            for weight in [.25,.5]:
                outputs=[]
                for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
                    f=fs[split];m=rows.class_name.eq(cls).to_numpy()
                    unique=f.assigned.eq(1)&f.rival_margin.ge(.1)&f.instances.le(1)
                    reliable=f.seg_stability.ge(.8)&f.mask_stability.ge(.75)&f.seg_touches.eq(0)&f.seg_area.between(.15,1.8)&f.ref_conf.ge(.65)
                    if rule=='consensus_rescue':gate=m&unique&f.families.ge(2)&f.consensus_min.ge(.65)&guard(split,rows,cls);target=np.clip((.85-f.consensus_mean.to_numpy())/.7,.05,.4)
                    elif rule=='consensus_penalty':gate=m&unique&f.families.ge(2)&f.consensus_max.between(.15,.45);target=np.full(len(rows),.85)
                    elif rule=='group_loser':gate=m&f.duplicate_loser.eq(1)&f.families.ge(2);target=np.full(len(rows),.9)
                    elif rule=='segmentation_rescue':gate=m&unique&reliable&f.seg_iou.ge(.7)&guard(split,rows,cls);target=np.full(len(rows),.3)
                    else:gate=m&unique&reliable&f.seg_iou.lt(.45)&f.seg_cover.lt(.55)&f.reference_iou.ge(.3);target=np.full(len(rows),.85)
                    rescue=rule.endswith('rescue');delta=np.minimum(target-base,0) if rescue else np.maximum(target-base,0)
                    score=base+weight*delta*np.asarray(gate,bool);outputs.append(score)
                register(f'{rule}_{cls}_w{weight}',*outputs,dict(type='independent_instance_evidence',rule=rule,class_name=cls,weight=weight))
