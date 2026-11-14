"""Small shape expert distinguishing a plate rim from a rectangular tray.

Manual train boxes only; grouped cross-day evaluation. Normalized image shape
features contain no frame IDs, coordinates, baseline score, or test annotations.
"""
from pathlib import Path
import sys,json,joblib
import cv2,numpy as np,pandas as pd
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v30'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import h,q,te,B,DAYS,align,base_oof,current
from features23 import image
from bounded_geometry import save
cv2.setNumThreads(2)

def descriptor(fid,b):
    wh=b[2:]-b[:2]
    xx,yy=np.meshgrid(np.linspace(b[0]-.12*wh[0],b[2]+.12*wh[0],96),
                     np.linspace(b[1]-.12*wh[1],b[3]+.12*wh[1],96))
    im=cv2.remap(image(fid),xx.astype('float32'),yy.astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype('float32')/255
    g=cv2.GaussianBlur(g,(3,3),.6);gx=cv2.Sobel(g,cv2.CV_32F,1,0);gy=cv2.Sobel(g,cv2.CV_32F,0,1)
    mag=np.hypot(gx,gy);mag/=max(float(np.quantile(mag,.9)),.02)
    hv=hog(g,orientations=8,pixels_per_cell=(12,12),cells_per_block=(1,1))
    # Rim curvature versus four straight sides, in normalized box coordinates.
    yy,xx=np.indices(g.shape);dx=xx-47.5;dy=yy-47.5;r=np.hypot(dx,dy)
    theta=(np.arctan2(dy,dx)+np.pi)/(2*np.pi)
    radial=[]
    for k in range(16):
        wedge=(theta>=k/16)&(theta<(k+1)/16)
        for lo,hi in [(12,25),(25,35),(35,45),(45,57)]:
            m=wedge&(r>=lo)&(r<hi);radial.append(float(mag[m].mean()))
    grid=cv2.resize(mag,(8,8),interpolation=cv2.INTER_AREA).ravel()
    return np.r_[hv,radial,grid,np.log(wh[0]/wh[1])].astype('float32')

def dataset():
    rng=np.random.default_rng(3001);out=[];xs=[]
    for ix,r in h[h.class_name.isin(['tray','empty_plate','full_plate'])].iterrows():
        b=r[B].to_numpy(float);wh=b[2:]-b[:2]
        for aug in range(7):
            bb=b if aug==0 else b+rng.uniform(-.07,.07,4)*np.tile(wh,2)
            xs.append(descriptor(r.frame_id,bb))
            out.append(dict(annotation=int(ix),frame_id=r.frame_id,day=r.day,class_name=r.class_name,
                            plate=int(r.class_name!='tray'),augmentation=aug))
    return pd.DataFrame(out),np.stack(xs)

def run():
    meta,x=dataset();meta.to_csv(A/'inputs/plate_tray_training.csv',index=False)
    np.save(A/'features/plate_tray_train.npy',x)
    qo=q[q.class_name.eq('tray')];tt=te[te.class_name.eq('tray')]
    xs={s:np.stack([descriptor(r.frame_id,np.array([getattr(r,c) for c in B])) for r in rows.itertuples()])
        for s,rows in [('oof',qo),('test',tt)]}
    models={'linear_c0.03':lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.03,class_weight='balanced',max_iter=1500)),
            'linear_c0.3':lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.3,class_weight='balanced',max_iter=1500)),
            'rbf_c1':lambda:make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',class_weight='balanced',probability=True,random_state=3001))}
    results=[];quality=[]
    for name,build in models.items():
        po=np.zeros(len(qo));pt=[]
        for day in DAYS:
            train=meta.day.eq(day).to_numpy();held=(~train)&meta.augmentation.eq(0).to_numpy()
            model=build();model.fit(x[train],meta.loc[train,'plate'])
            joblib.dump(model,A/f'models/plate_tray_{name}_fit{day}.joblib')
            pred=model.predict_proba(x[held])[:,1]
            quality.append(dict(model=name,fit_day=day,train_frames=meta.loc[train,'frame_id'].nunique(),
                train_trays=meta[train&meta.augmentation.eq(0)&meta.plate.eq(0)].shape[0],
                train_plates=meta[train&meta.augmentation.eq(0)&meta.plate.eq(1)].shape[0],
                held_boxes=int(held.sum()),balanced_accuracy=balanced_accuracy_score(meta.loc[held,'plate'],pred>.5),
                auc=roc_auc_score(meta.loc[held,'plate'],pred)))
            mask=qo.day.ne(day).to_numpy();po[mask]=model.predict_proba(xs['oof'][mask])[:,1]
            pt.append(model.predict_proba(xs['test'])[:,1])
        ptest=np.minimum(*pt) # Both independent day fits must support plate.
        fo=qo[['id']].assign(p_plate=po);ft=tt[['id']].assign(p_plate=ptest,p_fit25=pt[0],p_fit26=pt[1])
        fo.to_csv(A/f'features/plate_tray_{name}_oof.csv',index=False)
        ft.to_csv(A/f'features/plate_tray_{name}_test.csv',index=False)
        for cut in [.75,.9]:
            for weight in [.25,.5]:
                outs=[]
                for split,rows,base,pr in [('oof',q,base_oof,fo),('test',te,current,ft)]:
                    p=pr.set_index('id').p_plate.reindex(rows.id).fillna(0).to_numpy()
                    sem=align(ROOT/f'rascar_v16/semantic_both_{split}.csv',rows,'p_tray')
                    gate=rows.class_name.eq('tray').to_numpy()&(p>=cut)&(sem<.9)
                    outs.append(base+weight*np.maximum(.9-base,0)*gate)
                results.append(save(f'plate_{name}_p{cut}_w{weight}',*outs,dict(
                    type='plate_tray_shape_expert',threshold=cut,weight=weight,template_fit='opposite_day',test_labels_used=False)))
    pd.DataFrame(quality).to_csv(A/'results/plate_tray_quality.csv',index=False)
    pd.DataFrame(results).drop(columns='details').to_csv(A/'results/plate_tray_metrics.csv',index=False)
    print(pd.DataFrame(quality).to_string(index=False));print(pd.DataFrame(results).drop(columns='details').to_string(index=False))

if __name__=='__main__':run()
