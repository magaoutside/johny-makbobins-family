"""Twelve seat geometry with leave-frame-out training features and cross-day fitting."""
from common17 import *
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
import joblib

records=pd.read_csv(V16/'dino_records.csv')
heads=pd.read_csv(V16/'wevis_heads_references.csv')
bank=pd.read_csv(V16/'head_geometry_templates.csv')
head_arrays={fid:g[B].to_numpy() for fid,g in heads.groupby('frame_id')}

def table_map(fit):
    tables=h[h.class_name.eq('table')&(True if fit=='all' else h.day.eq(fit))]
    arrays=[g.sort_values('x1')[B].to_numpy() for _,g in tables.groupby('frame_id') if len(g)==6]
    return np.median(arrays,axis=0)

def seat_location(center,tables):
    w=tables[:,2]-tables[:,0];hh=tables[:,3]-tables[:,1];cx=(tables[:,0]+tables[:,2])/2
    anchors=np.vstack([np.c_[cx,tables[:,1]-.45*hh],np.c_[cx,tables[:,3]+.1*hh]])
    # The left table is seen obliquely; seat positions remain learned from all
    # available manual head coordinates rather than fixed test query boxes.
    scales=np.vstack([np.c_[w,hh],np.c_[w,hh]])
    dist=np.linalg.norm((anchors-center)/scales,axis=1)
    j=int(dist.argmin());return j,float(dist[j]),anchors[j],scales[j]

def extract(rows,fit,leave_frame_out=False):
    tables=table_map(fit);pool=bank if fit=='all' else bank[bank.day.eq(fit)]
    pool=pool.copy();pool['seat']=[seat_location(c,tables)[0] for c in pool[['head_x','head_y']].to_numpy()]
    out=[]
    for r in rows.itertuples():
        b=np.array([getattr(r,c) for c in B]);wh=np.maximum(b[2:]-b[:2],1);center=(b[:2]+b[2:])/2
        rec=dict(id=r.id,x=center[0]/1280,y=center[1]/720,w=wh[0]/1280,h=wh[1]/720,aspect=np.log(wh[0]/wh[1]),
            head_count=0,head_fraction=0.,head_relx=0.,head_rely=0.,seat_distance=9.,back=0.,table_cover=0.,table_ybottom=0.,
            template_n=0.,valid_fraction=0.,iou_mean=0.,iou_q25=0.,iou_q50=0.,iou_q75=0.,iou_q90=0.,iou_max=0.,
            height_ratio_q25=1.,height_ratio_q50=1.,height_ratio_q75=1.,top_offset=0.,bottom_offset=0.,template_distance=9.,height_cv=0.)
        for k in range(12):rec[f'seat{k}']=0.
        hh=head_arrays.get(r.frame_id,np.empty((0,4)));cc=(hh[:,:2]+hh[:,2:])/2
        inside=((cc>=b[:2])&(cc<=b[2:])).all(1);rec['head_count']=int(inside.sum())
        if not inside.any():out.append(rec);continue
        cand=np.flatnonzero(inside);target=np.array([center[0],b[1]+.2*wh[1]])
        j=cand[np.linalg.norm((cc[cand]-target)/wh,axis=1).argmin()];head=hh[j];hc=cc[j];hw=head[2:]-head[:2]
        seat,sd,anchor,scale=seat_location(hc,tables);table=tables[seat%6];tw=table[2:]-table[:2]
        rec.update(head_fraction=float(hw.prod()/wh.prod()),head_relx=float((hc[0]-b[0])/wh[0]),head_rely=float((hc[1]-b[1])/wh[1]),
            seat_distance=sd,back=float(seat<6),table_cover=float(np.maximum(0,np.minimum(table[2:],b[2:])-np.maximum(table[:2],b[:2])).prod()/wh.prod()),table_ybottom=float((b[3]-table[3])/tw[1]))
        rec[f'seat{seat}']=1.
        candidates=pool[pool.seat.eq(seat)&(pool.frame_id.ne(r.frame_id) if leave_frame_out else True)]
        if len(candidates)<2:out.append(rec);continue
        pc=candidates[['head_x','head_y']].to_numpy();dist=np.linalg.norm((pc-hc)/scale,axis=1)
        size_ratio=np.maximum(candidates.head_w.to_numpy(),1)/max(hw[0],1)
        dist=dist+.25*abs(np.log(size_ratio));ix=np.argsort(dist)[:7]
        ix=ix[dist[ix]<1.5]
        if len(ix)<2:out.append(rec);continue
        sub=candidates.iloc[ix];pb=sub[B].to_numpy();pc=pc[ix];weights=np.exp(-dist[ix]**2/(2*.55**2));weights/=weights.sum()
        translated=pb+np.tile(hc-pc,(1,2));ov=iou(b,translated)[0]
        def quantile(v,p):
            order=np.argsort(v);return float(v[order[min(np.searchsorted(np.cumsum(weights[order]),p),len(v)-1)]])
        th=translated[:,3]-translated[:,1]
        rec.update(template_n=len(ix),valid_fraction=float(weights@(ov>=.6)),iou_mean=float(weights@ov),iou_max=float(ov.max()),
            height_ratio_q25=float(wh[1]/quantile(th,.25)),height_ratio_q50=float(wh[1]/quantile(th,.5)),height_ratio_q75=float(wh[1]/quantile(th,.75)),
            top_offset=float(weights@((b[1]-translated[:,1])/th)),bottom_offset=float(weights@((translated[:,3]-b[3])/th)),
            template_distance=float(weights@dist[ix]),height_cv=float(np.std(th)/np.mean(th)))
        for level in [25,50,75,90]:rec[f'iou_q{level}']=quantile(ov,level/100)
        out.append(rec)
    return pd.DataFrame(out)

metadata={}
for fit in DAYS+['all']:
    tables=table_map(fit);bp=bank if fit=='all' else bank[bank.day.eq(fit)]
    counts=pd.Series([seat_location(c,tables)[0] for c in bp[['head_x','head_y']].to_numpy()]).value_counts().to_dict()
    metadata[fit]={'tables':tables.tolist(),'manual_head_templates_per_seat':{str(k):int(v) for k,v in counts.items()}}
(A/'seat_maps.json').write_text(json.dumps(metadata,indent=2))

features={};train_records={}
for fit,held in zip(DAYS,DAYS[::-1]):
    tr=records[records.day.eq(fit)&records.class_name.eq('person')&~records.kind.eq('query_test')].copy().reset_index(drop=True)
    # Hard-negative background crops are informative for semantics, but this
    # geometry branch only judges person-shaped queries with a detected head.
    x=extract(tr,fit,True);v=extract(q[q.class_name.eq('person')&q.day.eq(held)],fit);t=extract(te[te.class_name.eq('person')],fit)
    x.to_csv(A/f'seat_train_features_{fit}.csv',index=False);v.to_csv(A/f'seat_held_features_{held}.csv',index=False);t.to_csv(A/f'seat_test_features_fit_{fit}.csv',index=False)
    features[fit]=(x,v,t);train_records[fit]=tr
ft=extract(te,'all');ft.to_csv(A/'seat_features_test_all.csv',index=False)
fo=pd.concat([features[fit][1] for fit in DAYS]).set_index('id').reindex(q.id).fillna(0).reset_index();fo.to_csv(A/'seat_features_oof.csv',index=False)

for model_name in ['logreg','catboost','iou_regressor']:
    oof=np.zeros(len(q));test_predictions=[]
    for fit,held in zip(DAYS,DAYS[::-1]):
        x,v,t=features[fit];tr=train_records[fit];cols=[c for c in x if c!='id']
        weights=tr.weight.to_numpy().copy();weights[tr.kind.eq('query_train')]*=2
        eligible=x.head_count.gt(0)&x.template_n.ge(2)&tr.kind.ne('background')
        if model_name=='logreg':
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,class_weight='balanced'))
            model.fit(x.loc[eligible,cols],tr.loc[eligible,'error'],logisticregression__sample_weight=weights[eligible]);pv=model.predict_proba(v[cols])[:,1];pt=model.predict_proba(t[cols])[:,1]
        elif model_name=='catboost':
            model=CatBoostClassifier(iterations=350,depth=3,learning_rate=.03,l2_leaf_reg=15,random_seed=171,thread_count=2,verbose=False,allow_writing_files=False)
            model.fit(x.loc[eligible,cols],tr.loc[eligible,'error'],sample_weight=weights[eligible]);pv=model.predict_proba(v[cols])[:,1];pt=model.predict_proba(t[cols])[:,1]
        else:
            # Actual query IoUs come from same-frame manual GT, not the held day.
            target=tr.target_iou.to_numpy().copy()
            for j,r in tr[tr.kind.eq('query_train')].iterrows():
                gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq('person')][B].to_numpy();target[j]=float(iou(r[B],gt).max()) if len(gt) else 0.
            model=CatBoostRegressor(iterations=350,depth=3,learning_rate=.03,l2_leaf_reg=15,random_seed=171,thread_count=2,verbose=False,allow_writing_files=False)
            model.fit(x.loc[eligible,cols],target[eligible],sample_weight=weights[eligible]);pv=expit((.6-model.predict(v[cols]))/.08);pt=expit((.6-model.predict(t[cols]))/.08)
        joblib.dump(model,A/f'seat_{model_name}_{fit}.joblib')
        mask=q.class_name.eq('person')&q.day.eq(held);oof[mask]=pd.Series(pv,index=v.id).loc[q.loc[mask,'id']]
        test_predictions.append(pd.Series(pt,index=t.id).reindex(te.id).fillna(0).to_numpy())
    pt=np.mean(test_predictions,axis=0)
    pd.DataFrame({'id':q.id,'quality':oof}).to_csv(A/f'seat_{model_name}_oof.csv',index=False);pd.DataFrame({'id':te.id,'quality':pt}).to_csv(A/f'seat_{model_name}_test.csv',index=False)
    for weight in [.2,.4,.6]:
        outputs=[]
        for rows,base,pred,f in [(q,q.score.to_numpy(),oof,fo),(te,current,pt,ft)]:
            # Seat prior is advisory; do not apply to aisles, missing heads or wide bodies.
            m=rows.class_name.eq('person').to_numpy()&(f.head_count.to_numpy()==1)&(f.template_n.to_numpy()>=2)&(f.seat_distance.to_numpy()<1.2)&(rows.x2.sub(rows.x1)/rows.y2.sub(rows.y1)<1.6).to_numpy()
            s=base.copy();s[m]=(1-weight)*s[m]+weight*pred[m];outputs.append(s)
        register(f'seat_{model_name}_w{weight}',*outputs,details=dict(model=model_name,weight=weight,train_feature_templates='same fit day, leave query frame out',validation_templates='opposite day only',training='manual GT perturbations + real train queries'))

# Transparent rule ablation: distribution of allowed boxes at a seat, not a fixed extension.
for quantile in [50,75]:
    outputs=[]
    for rows,base,f in [(q,q.score.to_numpy(),fo),(te,current,ft)]:
        risk=expit((.6-f[f'iou_q{quantile}'].to_numpy())/.07)
        m=(f.head_count.to_numpy()==1)&(f.template_n.to_numpy()>=3)&(f.height_ratio_q50.to_numpy()<.8)&(f.top_offset.abs().to_numpy()<.2)&(f.seat_distance.to_numpy()<1.2)&rows.class_name.eq('person').to_numpy()
        outputs.append(base+.4*(1-base)*risk*m)
    register(f'seat_distribution_q{quantile}',*outputs,details=dict(quantile=quantile,weight=.4,height_ratio_max=.8,minimum_templates=3))
