"""Integrate IoU over empirical annotation uncertainty of frozen references.

The neighbours are fit-day detector/GT residuals, one per source object. They
describe how a detector box differs from annotation, not query correctness.
"""
from common27 import *
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

refs=pd.concat([pd.read_csv(ROOT/f'rascar_v15/v8/fused_{s}_references.csv') for s in ['train','test']],ignore_index=True)
refs=refs[refs.class_name.isin(CL)&refs.confidence.ge(.4)].copy()
def descriptor(box,confidence):
    box=np.asarray(box);wh=box[2:]-box[:2];cc=(box[:2]+box[2:])/2
    return np.r_[cc/[400,250],np.log(wh)/2,np.log(wh[0]/wh[1]),confidence]

def bank():
    out=[]
    for (fid,cls),g in refs.groupby(['frame_id','class_name']):
        gt=h[h.frame_id.eq(fid)&h.class_name.eq(cls)]
        if not len(gt):continue
        ov=iou(g[B],gt[B]);ii,jj=linear_sum_assignment(-ov)
        for i,j in zip(ii,jj):
            if ov[i,j]<.25:continue
            r=g.iloc[i];bb=r[B].to_numpy(float);wh=bb[2:]-bb[:2];truth=gt.iloc[j][B].to_numpy(float)
            delta=(truth-bb)/np.tile(wh,2)
            if abs(delta).max()>1.5:continue
            x=descriptor(bb,r.confidence)
            out.append(dict(frame_id=fid,day=gt.iloc[j].day,class_name=cls,gt_index=int(gt.index[j]),raw_iou=ov[i,j],
                **{f'f{k}':v for k,v in enumerate(x)},**{f'y{k}':v for k,v in enumerate(delta)}))
    return pd.DataFrame(out)

def features(rows,fit,mode,bank):
    out=[];samples={}
    for cls in CL:
        tr=bank[bank.day.eq(fit)&bank.class_name.eq(cls)]
        samples[cls]=(tr,tr.filter(regex='^f[0-9]').to_numpy(),tr.filter(regex='^y[0-9]').to_numpy())
    for r in rows.itertuples():
        cls=r.class_name;f=dict(id=r.id,probability=1.,best_iou=0.,support=0.,uncertainty=1.,conf=0.,ref_shift=0.)
        f.update({'ref_'+c:0. for c in B})
        g=refs[refs.frame_id.eq(r.frame_id)&refs.class_name.eq(cls)]
        tr,x,y=samples[cls]
        if not len(g) or len(tr)<8:out.append(f);continue
        box=np.array([getattr(r,c) for c in B]);ov=iou(box,g[B])[0]
        if ov.max()<.15:out.append(f);continue
        # Select by original geometry, never by optimistic corrected overlap.
        j=ov.argmax();ref=g.iloc[j];bb=ref[B].to_numpy(float);wh=bb[2:]-bb[:2]
        dist=np.linalg.norm(x-descriptor(bb,ref.confidence),axis=1)
        if mode=='global':dist[:]=0
        order=np.argsort(dist)[:min(20,len(tr))];weight=np.exp(-dist[order]**2/2)
        weight/=weight.sum();possible=bb+y[order]*np.tile(wh,2)
        overlap=iou(box,possible)[0];estimated=(possible*weight[:,None]).sum(0)
        prob=float((weight*(overlap<.6)).sum())
        f.update(probability=prob,best_iou=float(ov[j]),support=float(1/(weight**2).sum()),uncertainty=float(np.sqrt(weight@((overlap-weight@overlap)**2))),
            conf=float(ref.confidence),ref_shift=float(np.max(abs((estimated-bb)/np.tile(wh,2)))),**dict(zip(['ref_'+c for c in B],estimated)))
        out.append(f)
    return pd.DataFrame(out).set_index('id').loc[rows.id]

def main():
    tr=bank();tr.to_csv(A/'inputs/reference_residual_bank.csv',index=False)
    for mode in ['global','local']:
        of=[];tf=[]
        for fit,held in zip(DAYS,DAYS[::-1]):
            for split,rows in [('oof',q[q.day.eq(held)]),('test',te)]:
                f=features(rows,fit,mode,tr);f.to_csv(A/f'features/reference_posterior_{mode}_{split}_{fit}.csv')
                (of if split=='oof' else tf).append(f)
        fo=pd.concat(of).loc[q.id];ft=(tf[0]+tf[1])/2
        for cls in CL:
            for direction in ['rescue','penalty','both']:
                for weight in [.25,.5]:
                    scores=[]
                    for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
                        p=f.probability.to_numpy();supported=f.support.ge(8).to_numpy()&f.conf.ge(.6).to_numpy()
                        low=(p<.25)&(p<base)&guard(split,rows,cls);high=(p>.85)&(p>base)&f.best_iou.between(.15,.65).to_numpy()
                        if split=='test':low&=np.maximum(tf[0].probability,tf[1].probability).to_numpy()<.35;high&=np.minimum(tf[0].probability,tf[1].probability).to_numpy()>.75
                        gate=rows.class_name.eq(cls).to_numpy()&supported&(low if direction=='rescue' else high if direction=='penalty' else low|high)
                        scores.append(base+weight*gate*(p-base))
                    register(f'reference_posterior_{mode}_{cls}_{direction}_w{weight}',*scores,dict(type='empirical_detector_boundary_distribution',mode=mode,cls=cls,direction=direction,weight=weight))
if __name__=='__main__':main()
