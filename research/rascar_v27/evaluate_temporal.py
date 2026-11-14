"""Cross-day calibration and current-frame/temporal ablation, frozen inference."""
from common27 import *

def prepare():
    d=pd.read_csv(A/'gpu/temporal_references.csv')
    return d

def select(d,mode):
    d=d[d.confidence.ge(.4)].copy()
    if mode=='current':return d[d.seconds.eq(0)]
    valid=d.seconds.ne(0)&d.ncc.ge(.8)&d.tracked_fraction.ge(.5)&d.inlier_fraction.ge(.75)&d.fb_error.le(.8)&d.pixel_error.le(30)
    if mode=='temporal':return d[valid]
    return d[d.seconds.eq(0)|valid]

def padded(d,pad):
    d=d.copy();b=d[B].to_numpy();wh=b[:,2:]-b[:,:2];d[B]=b+np.c_[-wh*pad,wh*pad]
    return d

def padding(d,fit):
    manual=h[h.day.eq(fit)&h.class_name.eq('nude_hands')].copy();manual['id']=['manual_'+str(i) for i in manual.index]
    values=[]
    for pad in [-.15,-.075,0,.1,.2]:
        rr=padded(d,pad);scores=[]
        for r in manual.itertuples():
            ref=rr[rr.id.eq(r.id)]
            scores.append(float(iou([getattr(r,c) for c in B],ref[B]).max()) if len(ref) else 0.)
        values.append((float(np.mean(scores)),pad))
    return max(values)[1],values

def features(rows,d,pad):
    d=padded(d,pad);out=[]
    for r in rows.itertuples():
        f=dict(id=r.id,best_iou=0.,confidence=0.,turns=0,frames=0,coherence=0.,ncc=0.,seconds=0.,reference_count=0)
        f.update({'ref_'+c:0. for c in B})
        g=d[d.id.eq(r.id)]
        if not len(g):out.append(f);continue
        b=np.array([getattr(r,c) for c in B]);ov=iou(b,g[B])[0];j=ov.argmax();ref=g.iloc[j]
        support=g.loc[ov>=.6];boxes=support[B].to_numpy();best=ref[B].to_numpy(float)
        f.update(best_iou=float(ov[j]),confidence=float(ref.confidence),turns=int(support.turn.nunique()),frames=int(support.source_frame.nunique()),
            coherence=float(np.median(iou(best,boxes))) if len(boxes) else 0.,ncc=float(ref.ncc),seconds=float(ref.seconds),reference_count=len(g),
            **dict(zip(['ref_'+c for c in B],best)))
        out.append(f)
    return pd.DataFrame(out).set_index('id').loc[rows.id]

def main():
    d=prepare();benchmark=[];cal=[]
    pool=pd.read_csv(V22/'inputs/reviewed_pool_hands.csv')
    for mode in ['current','temporal','combined']:
        rr=select(d,mode);of=[];tf=[]
        for fit,held in zip(DAYS,DAYS[::-1]):
            pad,grid=padding(rr,fit);cal.append(dict(mode=mode,fit=fit,pad=pad,grid=grid))
            for split,rows in [('oof',q[q.day.eq(held)]),('test',te)]:
                f=features(rows,rr,pad);f.to_csv(A/f'features/temporal_{mode}_{split}_{fit}.csv');(of if split=='oof' else tf).append(f)
            manual=h[h.day.eq(held)&h.class_name.eq('nude_hands')].copy();manual['id']=['manual_'+str(i) for i in manual.index]
            for source,rows in [('official_manual',manual),('reviewed_pool',pool[pool.day.eq(held)])]:
                f=features(rows,rr,pad).reset_index();f['source']=source;f['held']=held;f['mode']=mode;benchmark.append(f)
        fo=pd.concat(of).loc[q.id];ft=(tf[0]+tf[1])/2
        for minimum in [.6,.7]:
            for support in [1,2]:
                for weight in [.35,.6]:
                    outputs=[]
                    for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
                        gate=rows.class_name.eq('nude_hands').to_numpy()&guard(split,rows,'nude_hands')&f.best_iou.ge(minimum).to_numpy()
                        count=f.turns if mode=='current' else f.frames
                        gate&=count.ge(support).to_numpy()&f.coherence.ge(.65).to_numpy()
                        if split=='test':
                            gate&=np.minimum(tf[0].best_iou,tf[1].best_iou).to_numpy()>=minimum
                            countcol='turns' if mode=='current' else 'frames'
                            gate&=np.minimum(tf[0][countcol],tf[1][countcol]).to_numpy()>=support
                        target=np.clip((.9-f.best_iou.to_numpy())/.6,0,1)
                        outputs.append(base-weight*gate*np.maximum(base-target,0))
                    register(f'temporal_{mode}_iou{minimum}_support{support}_w{weight}',*outputs,
                        dict(type='WiLoR_TTA_and_optical_flow',mode=mode,min_iou=minimum,support=support,weight=weight,padding_fit='opposite day only',temporal_frames='unlabeled neighbouring frames of inference day, ±65sec'))
    pd.concat(benchmark).to_csv(A/'results/temporal_detection_benchmark.csv',index=False)
    (A/'results/temporal_calibration.json').write_text(json.dumps(cal,indent=2))
if __name__=='__main__':main()
