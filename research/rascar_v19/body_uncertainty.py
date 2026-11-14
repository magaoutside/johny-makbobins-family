"""Frame-cross-fitted body residuals; uncertainty replaces hard point-IoU cut."""
from common19 import *
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
x=pd.read_csv(V18/'person_body_training_features.csv');target=pd.read_csv(V18/'person_body_training_targets.csv')
xq=pd.read_csv(V18/'person_body_query_features_oof.csv');xt=pd.read_csv(V18/'person_body_query_features_test.csv')
mq=pd.read_csv(V18/'person_body_query_heads_oof.csv');mt=pd.read_csv(V18/'person_body_query_heads_test.csv')
def probabilities(rows,features,matches,model,residuals):
    predicted=model.predict(features);out=[]
    for i,r in enumerate(rows.itertuples()):
        center=matches.iloc[i][['hx','hy']].to_numpy(float);size=matches.iloc[i][['hw','hh']].to_numpy(float)
        refs=(predicted[i]+residuals)*np.tile(size,2)+np.tile(center,2)
        valid=((refs[:,2:]-refs[:,:2])>2).all(1)&(refs[:,0]>-200)&(refs[:,1]>-100)&(refs[:,2]<1480)&(refs[:,3]<900)
        refs=refs[valid];b=np.array([getattr(r,c) for c in B]);ov=iou(b,refs)[0]
        out.append(dict(id=r.id,body_error_probability=float((ov<.6).mean()) if len(refs) else 0.,body_iou_median=float(np.median(ov)) if len(refs) else 0.,residual_support=len(refs)))
    return pd.DataFrame(out)
fo=[];ft=[]
for fit,held in zip(DAYS,DAYS[::-1]):
    mask=target.day.eq(fit);res=[]
    for fid in target.loc[mask,'frame_id'].unique():
        val=mask&target.frame_id.eq(fid);train=mask&~target.frame_id.eq(fid)
        model=make_pipeline(StandardScaler(),Ridge(alpha=100));model.fit(x[train],target.loc[train,B]);pred=model.predict(x[val])
        res.extend((target.loc[val,B].to_numpy()-pred).tolist())
    res=np.asarray(res);pd.DataFrame(res,columns=B).to_csv(A/f'body_residuals_{fit}.csv',index=False)
    model=joblib.load(V18/f'person_body_ridge_{fit}.joblib');m=q.day.eq(held)
    f=probabilities(q[m].reset_index(drop=True),xq[m].reset_index(drop=True),mq[m].reset_index(drop=True),model,res);fo.append(f)
    ft.append(probabilities(te,xt,mt,model,res))
fo=pd.concat(fo).set_index('id').loc[q.id].reset_index();ft=pd.concat(ft).groupby('id',sort=False).mean().loc[te.id].reset_index()
fo.to_csv(A/'body_uncertainty_oof.csv',index=False);ft.to_csv(A/'body_uncertainty_test.csv',index=False)
for weight in [.25,.5,.75,1.]:
    outputs=[]
    for rows,base,test,f in [(q,q.score.to_numpy(),False,fo),(te,current,True,ft)]:
        split='test' if test else 'oof';point=align(V18/f'person_body_ridge_{split}.csv',rows);seat=align(V17/('seat_features_test_all.csv' if test else 'seat_features_oof.csv'),rows)
        eligible=point.eligible.to_numpy(bool)&(point.height_ratio.to_numpy()<.85)&(point.top_error.to_numpy()<.2)&point.width_ratio.between(.65,1.5).to_numpy()
        eligible&=seat.back.to_numpy(bool)&(seat.head_count.to_numpy()==1)&(seat.template_n.to_numpy()>=3)&(seat.seat_distance.to_numpy()<1.2)&(rows.y2.to_numpy()<710)&(rows.x1.to_numpy()>5)&(rows.x2.to_numpy()<1275)&(f.body_error_probability.to_numpy()>=.5)
        base16=old.old.current if test else old.old.q.score.to_numpy()
        candidate=base16+weight*(1-base16)*f.body_error_probability.to_numpy()
        outputs.append(np.where(eligible,np.maximum(base,candidate),base))
    register(f'body_uncertainty_w{weight}',*outputs,details=dict(weight=weight,uncertainty='leave-one-frame-out residuals on fitting day only',height_ratio_max=.85,error_probability_min=.5,combination='max related corrections'))
