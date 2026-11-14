"""Continuous 0.5 contours instead of rounding every box edge to a patch border.

No query coordinates or targets are used to refine a component. The reference is
decoded first, then matched to the query as in V31.
"""
from common55 import *
from train import score_dense
import cv2,argparse
from sklearn.metrics import average_precision_score as AP

def decode(p):
    _,labels,stats,_=cv2.connectedComponentsWithStats((p>=.5).astype(np.uint8),8);boxes=[]
    padded=np.pad(p,1)
    for j,(x,y,w,h,area) in enumerate(stats[1:],1):
        if area<4:continue
        c=labels==j;yy,xx=np.where(c);cur=p[yy,xx]
        # Between the centers of a foreground cell and its adjacent background cell.
        xmin=[];xmax=[];ymin=[];ymax=[]
        for dy,dx,axis,direction,target in [(0,-1,xx,-1,xmin),(0,1,xx,1,xmax),(-1,0,yy,-1,ymin),(1,0,yy,1,ymax)]:
            neighbor=padded[yy+1+dy,xx+1+dx];edge=neighbor<.5
            distance=np.clip((cur[edge]-.5)/np.maximum(cur[edge]-neighbor[edge],1e-8),0,1)
            target.extend((axis[edge]+.5+direction*distance).tolist())
        if not all([xmin,xmax,ymin,ymax]):continue
        b=np.clip([min(xmin),min(ymin),max(xmax),max(ymax)],0,p.shape[-1])
        if c[:,0].any():b[0]=0
        if c[0].any():b[1]=0
        if c[:,-1].any():b[2]=p.shape[-1]
        if c[-1].any():b[3]=p.shape[-1]
        boxes.append(b)
    return boxes

def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',default=['control450']);p.add_argument('--seeds',nargs='+',type=int,default=[3101]);p.add_argument('--tag',default='initial');args=p.parse_args()
    q,t,h=inputs();bo,bt=incumbent(q,t);v31o,v31t=baseline(q,t)
    rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv').set_index('id');results=[]
    for name in args.names:
        evidence=[]
        for fit in DAYS:
            maps=[];ids=None
            for seed in args.seeds:
                d=np.load(A/f'features/{name}_s{seed}_fit{fit[-2:]}_maps.npz');maps.append(d['maps']);ids=d['ids']
            prob=np.mean(maps,axis=0)
            for k,identifier in enumerate(ids):
                r=rec.loc[identifier];crop=r[CB].to_numpy(float)
                boxes=decode(prob[k,CL.index(r.class_name)])
                bb=np.array(boxes).reshape(-1,4)/32*np.tile(crop[2:]-crop[:2],2)+np.tile(crop[:2],2)
                overlap=float(iou(r[B],bb).max()) if len(bb) else np.nan
                evidence.append(dict(id=identifier,split=r.split,fit_day=fit,overlap=overlap))
        e=pd.DataFrame(evidence);e.to_csv(A/f'features/{name}_subpixel_{args.tag}_evidence.csv',index=False)
        oo=e[e.split.eq('oof')].set_index('id').loc[q.id,'overlap'].to_numpy()
        tt=np.stack([g.set_index('id').loc[t.id,'overlap'].to_numpy() for _,g in e[e.split.eq('test')].groupby('fit_day')])
        s0,t0=score_dense(oo,tt,q,t)
        for scope in ['all','person','nude_hands']:
            om=q.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(q),bool);tm=t.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(t),bool)
            for fraction in [.5,1.]:
                s,z=bo.copy(),bt.copy();s[om]+=fraction*(s0-v31o)[om]*np.where(q.class_name.eq('person'),1/1.2,1)[om];z[tm]+=fraction*(t0-v31t)[tm]*np.where(t.class_name.eq('person'),1/1.2,1)[tm]
                label=f'{name}_subpixel_{args.tag}_{scope}_replace{int(fraction*100)}'
                q.assign(score=s).to_csv(A/f'results/oof/{label}.csv',index=False);t[['id']].assign(score=z).to_csv(A/f'results/submissions/{label}.csv',index=False)
                results.append(dict(name=label,recipe=name,scope=scope,fraction=fraction,**metrics(q,s),delta=AP(q.error,s)-AP(q.error,bo),changed_test=int((abs(z-bt)>1e-9).sum())))
    d=pd.DataFrame(results).sort_values('ap',ascending=False);d.to_csv(A/f'results/subpixel_experiments_{args.tag}.csv',index=False);print(d.to_string(index=False))

if __name__=='__main__':main()
