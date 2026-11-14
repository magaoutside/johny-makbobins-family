"""Fit the detector's anatomical-to-annotation boundary error, not the query.

All targets are official manual hands from the fit day. Per-frame weighting
prevents repeated pose views from multiplying the effective sample size.
"""
from common27 import *
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import joblib

MODELS=['rtmw','rtmwx_bgr','rtmwx_rgb']
def load_refs(model):
    d=pd.read_csv(ROOT/f'rascar_v25/gpu/results/{model}/keypoints.csv');out=[]
    for i,r in enumerate(d.itertuples()):
        pts=np.asarray(json.loads(r.xy));conf=np.asarray(json.loads(r.confidence))
        if (conf>.2).sum()<15:continue
        lo=pts.min(0);hi=pts.max(0);wh=hi-lo
        if min(wh)<3 or max(wh)>150:continue
        norm=(pts-lo)/wh
        # Normalized landmark layout includes the wrist and all five fingers.
        x=np.r_[norm.ravel(),np.log(wh),lo/[1280,720],np.quantile(conf,[.1,.5,.9])]
        out.append(dict(ref_id=i,frame_id=r.frame_id,turn=r.turn,confidence=float(np.median(conf)),
                    **dict(zip(B,np.r_[lo,hi])),**{f'f{k}':v for k,v in enumerate(x)}))
    return pd.DataFrame(out)

def fit_pairs(d):
    out=[]
    for fid,g in d.groupby('frame_id'):
        gt=h[h.frame_id.eq(fid)&h.class_name.eq('nude_hands')]
        if not len(gt):continue
        ov=iou(g[B],gt[B]);best=ov.argmax(1)
        for k,(idx,r) in enumerate(g.iterrows()):
            if ov[k,best[k]]<.25:continue
            box=r[B].to_numpy(float);truth=gt.iloc[best[k]][B].to_numpy(float)
            offset=(truth-box)/np.tile(box[2:]-box[:2],2)
            if abs(offset).max()>1.5:continue
            out.append(dict(ref_id=r.ref_id,frame_id=fid,day=gt.iloc[0].day,gt_index=int(gt.index[best[k]]),
                       original_iou=float(ov[k,best[k]]),**{f'y{j}':v for j,v in enumerate(offset)}))
    return pd.DataFrame(out)

def estimator(kind):
    if kind=='ridge':return make_pipeline(StandardScaler(),Ridge(alpha=100))
    return ExtraTreesRegressor(n_estimators=160,min_samples_leaf=12,max_depth=7,max_features=.65,random_state=2701,n_jobs=2)

def fitted(d,pairs,fit,kind):
    tr=pairs[pairs.day.eq(fit)].copy();x=d.set_index('ref_id').loc[tr.ref_id].filter(regex='^f[0-9]+$').to_numpy()
    y=tr[[f'y{i}' for i in range(4)]].to_numpy();groups=tr.frame_id.to_numpy()
    weights=tr.groupby(['frame_id','gt_index']).ref_id.transform('size').rdiv(1).to_numpy()
    def train(x,y,w):
        m=estimator(kind)
        m.fit(x,y,**({'ridge__sample_weight':w} if kind=='ridge' else {'sample_weight':w}))
        return m
    oof=np.empty_like(y)
    for a,b in GroupKFold(min(4,len(set(groups)))).split(x,y,groups):oof[b]=train(x[a],y[a],weights[a]).predict(x[b])
    residual=y-oof
    # One residual per physical object; retained for the uncertainty experiment.
    residual=pd.DataFrame(residual).assign(gt=tr.gt_index.to_numpy()).groupby('gt').median().to_numpy()
    model=train(x,y,weights)
    return model,residual,dict(fit=fit,kind=kind,rows=len(tr),frames=tr.frame_id.nunique(),objects=tr.gt_index.nunique(),inner_mae=float(np.mean(abs(y-oof))))

def predict_refs(d,model):
    x=d.filter(regex='^f[0-9]+$').to_numpy();b=d[B].to_numpy();wh=b[:,2:]-b[:,:2]
    delta=model.predict(x).clip(-.65,.65);bb=b+delta*np.tile(wh,(1,2))
    out=d[['ref_id','frame_id','turn','confidence']].copy();out[B]=bb
    return out[(bb[:,2:]-bb[:,:2]).min(1)>3]

def features(rows,refs,residual):
    out=[];groups={fid:g for fid,g in refs.groupby('frame_id')}
    for r in rows.itertuples():
        f=dict(id=r.id,best_iou=0.,p_correct=0.,confidence=0.,uncertainty=1.,turns=0.)
        f.update({'ref_'+c:0. for c in B})
        if r.class_name!='nude_hands' or r.frame_id not in groups:out.append(f);continue
        g=groups[r.frame_id];b=np.array([getattr(r,c) for c in B]);rb=g[B].to_numpy()
        ov=iou(b,rb)[0];j=ov.argmax();best=rb[j];wh=best[2:]-best[:2]
        possible=best+residual*np.tile(wh,2)
        valid=(possible[:,2:]-possible[:,:2]).min(1)>1
        potential=iou(b,possible[valid])[0]
        f.update(best_iou=float(ov[j]),p_correct=float(np.mean(potential>=.6)),confidence=float(g.iloc[j].confidence),
                 uncertainty=float(potential.std()),turns=int(g.loc[ov>.6,'turn'].nunique()),**dict(zip(['ref_'+c for c in B],best)))
        out.append(f)
    return pd.DataFrame(out).set_index('id').loc[rows.id]

def main():
    ledger=[];bench=[]
    for modelname in MODELS:
        d=load_refs(modelname);pairs=fit_pairs(d);pairs.to_csv(A/f'inputs/{modelname}_calibration_pairs.csv',index=False)
        for kind in ['ridge','extra']:
            of=[];tf=[]
            for fit,held in zip(DAYS,DAYS[::-1]):
                model,residual,log=fitted(d,pairs,fit,kind);ledger.append(dict(model=modelname,**log))
                joblib.dump(dict(model=model,residual=residual,log=log),A/f'models/{modelname}_{kind}_{fit}.joblib')
                refs=predict_refs(d,model);refs.to_csv(A/f'features/{modelname}_{kind}_refs_{fit}.csv',index=False)
                for split,rows in [('oof',q[q.day.eq(held)]),('test',te)]:
                    f=features(rows,refs,residual);f.to_csv(A/f'features/{modelname}_{kind}_{split}_{fit}.csv')
                    (of if split=='oof' else tf).append(f)
                for r in h[h.day.eq(held)&h.class_name.eq('nude_hands')].itertuples():
                    g=refs[refs.frame_id.eq(r.frame_id)];raw=d[d.frame_id.eq(r.frame_id)];box=[getattr(r,c) for c in B]
                    bench.append(dict(model=modelname,kind=kind,fit=fit,held=held,gt_index=r.Index,frame_id=r.frame_id,
                        raw_iou=float(iou(box,raw[B]).max()) if len(raw) else 0,
                        corrected_iou=float(iou(box,g[B]).max()) if len(g) else 0))
            fo=pd.concat(of).loc[q.id];ft=(tf[0]+tf[1])/2
            for signal in ['iou','probability']:
                for cutoff in [.65,.75]:
                    for weight in [.3,.6]:
                        scores=[]
                        for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
                            value=f.best_iou.to_numpy() if signal=='iou' else f.p_correct.to_numpy()
                            gate=rows.class_name.eq('nude_hands').to_numpy()&guard(split,rows,'nude_hands')&(value>=cutoff)
                            if split=='test':
                                col='best_iou' if signal=='iou' else 'p_correct'
                                gate&=np.minimum(tf[0][col],tf[1][col]).to_numpy()>=cutoff
                            target=np.clip((.9-f.best_iou.to_numpy())/.6,0,1) if signal=='iou' else 1-f.p_correct.to_numpy()
                            scores.append(base-weight*np.maximum(base-target,0)*gate)
                        register(f'landmark_{modelname}_{kind}_{signal}_{cutoff}_w{weight}',*scores,
                            dict(type='annotation_calibrated_landmarks',model=modelname,kind=kind,signal=signal,cutoff=cutoff,weight=weight))
    pd.DataFrame(ledger).to_csv(A/'results/landmark_training.csv',index=False)
    pd.DataFrame(bench).to_csv(A/'results/landmark_boundary_benchmark.csv',index=False)
if __name__=='__main__':main()
