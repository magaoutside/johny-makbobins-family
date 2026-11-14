from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP

ROOT=Path(__file__).resolve().parent.parent
A=ROOT/'rascar_v37'
B=['x1','y1','x2','y2']; DAYS=['2026-10-25','2026-10-26']; CL=['person','nude_hands','cup','tray']
for name in ['inputs','features','models','audit','logs','results/oof','results/submissions']:
    (A/name).mkdir(parents=True,exist_ok=True)
q=pd.read_csv(ROOT/'rascar_v31/results/oof/linear_mask_cut0.5_all_two_sided_w0.25.csv')
te=pd.read_csv(ROOT/'rascar_v22/inputs/test_queries.csv')
te['day']=pd.to_datetime(te.frame_id.str[-15:-7],format='%Y%m%d').dt.strftime('%Y-%m-%d')
h=pd.read_csv(ROOT/'rascar_v22/inputs/manual_boxes.csv')
ROWS={'oof':q,'test':te}
BASE={'oof':q.score.to_numpy(),'test':pd.read_csv(ROOT/'rascar_v31/submission_v31_dense_oof098784.csv').set_index('id').loc[te.id].score.to_numpy()}

def iou(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    return inter/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None]-inter,1e-9)

def metric(rows,s):
    out={'pooled_ap':float(AP(rows.error,s))}
    for col in ['day','class_name']:
        for k,group in rows.groupby(col):
            mask=rows[col].eq(k);out[k]=float(AP(rows.loc[mask,'error'],np.asarray(s)[mask]))
    return out

def save(name,scores,description):
    changes=[]
    for split,rows in ROWS.items():
        s=scores[split];assert len(s)==len(rows) and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
        out=rows.assign(score=s) if split=='oof' else rows[['id']].assign(score=s)
        path=A/f'results/oof/{name}.csv' if split=='oof' else A/f'results/submissions/submission_{name}.csv'
        out.to_csv(path,index=False)
        d=rows.assign(split=split,v31_score=BASE[split],new_score=s,delta=s-BASE[split]);changes.append(d[abs(d.delta)>1e-10])
    pd.concat(changes).to_csv(A/f'audit/{name}_changes.csv',index=False)
    m=dict(experiment=name,**metric(q,scores['oof']),description=description,public_ap=None,submitted=False)
    m.update({f'changed_{s}':int((abs(scores[s]-BASE[s])>1e-10).sum()) for s in ROWS})
    print('RESULT',name,round(m['pooled_ap'],8),m['changed_oof'],m['changed_test'],flush=True)
    return m

def gallery(rows,path,label_columns=(),gt=False):
    from PIL import Image,ImageDraw
    import cv2
    for start in range(0,len(rows),12):
        batch=rows.iloc[start:start+12];canvas=Image.new('RGB',(1600,1200),'#202020');dr=ImageDraw.Draw(canvas)
        for j,(_,r) in enumerate(batch.iterrows()):
            b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2
            lo=np.maximum(0,np.floor(cc-1.2*wh)).astype(int);hi=np.minimum([1280,720],np.ceil(cc+1.2*wh)).astype(int)
            raw=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720))
            im=Image.fromarray(raw[lo[1]:hi[1],lo[0]:hi[0],::-1]);scale=min(395/im.width,322/im.height)
            im=im.resize((int(im.width*scale),int(im.height*scale)));x=j%4*400;y=j//4*400;canvas.paste(im,(x,y+72))
            def box(bb,color):
                bb=(np.asarray(bb)-np.r_[lo,lo])*scale+[x,y+72,x,y+72]
                bb[[0,2]]=np.clip(bb[[0,2]],x,x+im.width-1);bb[[1,3]]=np.clip(bb[[1,3]],y+72,y+72+im.height-1)
                if bb[2]>bb[0] and bb[3]>bb[1]:dr.rectangle(tuple(bb),outline=color,width=2)
            if gt:
                for bb in h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy():
                    if iou(b,bb)[0,0]>.01:box(bb,'#33ff88')
            box(b,'#ff5050')
            dr.text((x+3,y+2),f'{r.get("review_index",start+j)} {r.class_name} {r.id[:10]}',fill='white')
            for k,col in enumerate(label_columns):dr.text((x+3,y+18+14*k),f'{col}: {r[col]}',fill='white')
        canvas.save(Path(str(path)+f'_{start//12+1}.jpg'))
