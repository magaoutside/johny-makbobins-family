"""Predict hand boundary offsets instead of classifying a crop as erroneous."""
from common24 import *
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
import joblib

def targets(rows):
    ys=[];keep=[]
    for r in rows.itertuples():
        b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2]
        gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq('nude_hands')][B].to_numpy()
        if not len(gt):keep.append(False);ys.append(np.zeros(4));continue
        ov=iou(b,gt)[0];best=gt[ov.argmax()];delta=(best-b)/np.tile(wh,2)
        eligible=ov.max()>=.15 and abs(delta).max()<1.8 and r.kind not in ['merge','wrong_class']
        keep.append(eligible);ys.append(delta)
    return np.asarray(ys),np.asarray(keep)

def predictions(model,x,rows):
    b=rows[B].to_numpy();scale=np.tile(b[:,2:]-b[:,:2],(1,2))
    offsets=np.array([tree.predict(x.to_numpy()) for tree in model.estimators_]);boxes=b[None]+offsets*scale[None]
    lo=np.minimum(boxes[:,:,:2],boxes[:,:,2:]-1);hi=np.maximum(boxes[:,:,2:],lo+1);boxes=np.concatenate([lo,hi],axis=2)
    inter=np.maximum(0,np.minimum(b[None,:,2:],boxes[:,:,2:])-np.maximum(b[None,:,:2],boxes[:,:,:2])).prod(2)
    ov=inter/np.maximum((b[:,2:]-b[:,:2]).prod(1)[None]+(boxes[:,:,2:]-boxes[:,:,:2]).prod(2)-inter,1)
    bb=np.median(boxes,axis=0)
    return pd.DataFrame(dict(id=rows.id,probability=(ov<.6).mean(0),iou_q25=np.quantile(ov,.25,axis=0),iou_q75=np.quantile(ov,.75,axis=0),
        uncertainty=np.std(offsets,axis=0).mean(1),**dict(zip(['ref_'+v for v in B],bb.T))))

if __name__=='__main__':
    tr=pd.read_csv(A/'inputs/boundary_training.csv');tr=tr[tr.class_name.eq('nude_hands')].reset_index(drop=True)
    y,eligible=targets(tr);tr=tr[eligible].reset_index(drop=True);y=y[eligible]
    tr.assign(target_x1=y[:,0],target_y1=y[:,1],target_x2=y[:,2],target_y2=y[:,3]).to_csv(A/'inputs/hand_offset_training.csv',index=False)
    x=align(A/'features/boundary_train.csv',tr);ledger=[]
    for family in ['extra','forest']:
        os=[];ts=[]
        for fit,held in zip(DAYS,DAYS[::-1]):
            m=tr.day.eq(fit).to_numpy();model=(ExtraTreesRegressor if family=='extra' else RandomForestRegressor)(n_estimators=250,max_depth=14,min_samples_leaf=8,max_features=.7,n_jobs=2,random_state=2401)
            model.fit(x.loc[m],y[m],sample_weight=tr.loc[m,'weight'])
            joblib.dump(dict(model=model,columns=x.columns.tolist(),fit_day=fit,training_frames=sorted(tr.loc[m,'frame_id'].unique())),A/f'models/refine_hand_{family}_{fit}.joblib',compress=3)
            for split,rows in [('oof',q[q.day.eq(held)]),('test',te)]:
                f=align(A/f'features/boundary_{split}.csv',rows);d=predictions(model,f,rows)
                d.to_csv(A/f'features/offset_{family}_{split}_{fit}.csv',index=False)
                (os if split=='oof' else ts).append(d)
            ledger.append(dict(family=family,fit=fit,held=held,rows=int(m.sum()),frames=tr.loc[m,'frame_id'].nunique()))
            print('REFINE',ledger[-1],flush=True)
        fo=pd.concat(os).set_index('id').loc[q.id];tt=[d.set_index('id').loc[te.id] for d in ts];ft=(tt[0]+tt[1])/2
        for mode in ['rescue','penalty','both']:
            for weight in [.25,.5]:
                scores=[]
                for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
                    p=f.probability.to_numpy();lo=(p<.25)&f.iou_q25.gt(.65).to_numpy()&guard(split,rows,'nude_hands')
                    hi=(p>.85)&f.iou_q75.lt(.5).to_numpy()
                    if split=='test':lo&=np.maximum(tt[0].probability,tt[1].probability).to_numpy()<.35;hi&=np.minimum(tt[0].probability,tt[1].probability).to_numpy()>.75
                    gate=rows.class_name.eq('nude_hands').to_numpy()&(lo if mode=='rescue' else hi if mode=='penalty' else lo|hi)
                    delta=np.minimum(p-base,0) if mode=='rescue' else np.maximum(p-base,0) if mode=='penalty' else p-base
                    scores.append(base+weight*delta*gate)
                register(f'hand_offsets_{family}_{mode}_w{weight}',*scores,dict(type='hand_boundary_offset_regression',model=family,mode=mode,weight=weight))
    pd.DataFrame(ledger).to_csv(A/'results/offset_training_ledger.csv',index=False)
