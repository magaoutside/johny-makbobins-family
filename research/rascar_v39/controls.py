"""Held-day official-object localization and class-absent crop controls.

These are development diagnostics, not competition AP or fresh independent labels.
Positive crops are averaged per GT object before reporting; empty crops per frame.
"""
from set_detector import *
from refit import Head as MaskHead
import cv2

def mask_boxes(prob,r):
    k=CL.index(r.class_name);p=prob[k];n,lab,stats,_=cv2.connectedComponentsWithStats((p>=.5).astype(np.uint8),8)
    cb=r[CB].to_numpy(float);sz=cb[2:]-cb[:2];out=[]
    for j in range(1,n):
        x,y,w,hh,area=stats[j]
        if area<4:continue
        b=np.array([x,y,x+w,y+hh])/32;internal=np.array([cb[0]>0,cb[1]>0,cb[2]<1280,cb[3]<720]);edge=np.r_[b[:2]<.02,b[2:]>.98]&internal
        if edge.any():continue
        out.append(dict(class_name=r.class_name,confidence=float(p[lab==j].mean()),crop_edge=False,**dict(zip(B,b*np.r_[sz,sz]+np.r_[cb[:2],cb[:2]]))))
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-run',default='set');args=parser.parse_args();run=args.model_run;data=Data();rec=data.rec;positives=[];negative=[]
    for fit in DAYS:
        original=rec[rec.bank.eq(0)&rec.split.eq('manual')&rec.day.ne(fit)&rec.class_name.isin(CL)]
        for seed in SEEDS:
            pred=pd.read_csv(A/f'features/{run}_fit{fit[-2:]}_seed{seed}_boxes.csv');pred=pred[pred.confidence.ge(.5)&~pred.crop_edge]
            for _,r in original.iterrows():
                refs=pred[pred.record_id.eq(r.id)&pred.class_name.eq(r.class_name)];v=iou(r[B],refs[B]).ravel()
                positives.append(dict(kind=run,fit_day=fit,seed=seed,id=r.id,origin_id=r.origin_id,frame_id=r.frame_id,class_name=r.class_name,coverage=len(v)>0,overlap=float(v.max()) if len(v) else 0.))
        old=pd.read_csv(ROOT/f'rascar_v31/features/linear_mask_fit{fit[-2:]}_boxes.csv');old=old[old.threshold.eq(.5)]
        for _,r in original.iterrows():
            refs=old[old.record_id.eq(r.id)&old.class_name.eq(r.class_name)]
            cb=r[CB].to_numpy(float);sz=cb[2:]-cb[:2];normalized=(refs[B].to_numpy()-np.r_[cb[:2],cb[:2]])/np.r_[sz,sz]
            internal=np.array([cb[0]>0,cb[1]>0,cb[2]<1280,cb[3]<720])
            edge=(np.c_[normalized[:,:2]<.02,normalized[:,2:]>.98]&internal).any(1)
            refs=refs.loc[~edge];v=iou(r[B],refs[B]).ravel()
            positives.append(dict(kind='v31',fit_day=fit,seed=3101,id=r.id,origin_id=r.origin_id,frame_id=r.frame_id,class_name=r.class_name,coverage=len(v)>0,overlap=float(v.max()) if len(v) else 0.))
        ids=rec.index[rec.split.eq('empty')&rec.day.ne(fit)].to_numpy()
        for kind,seeds in [(run,SEEDS),('v31',[3101])]:
            for seed in seeds:
                model=MaskHead() if kind=='v31' else SetHead();path=ROOT/f'rascar_v31/models/linear_mask_fit{fit[-2:]}.pt' if kind=='v31' else A/f'models/{run}_fit{fit[-2:]}_seed{seed}.pt'
                model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
                for off in range(0,len(ids),16):
                    ix=ids[off:off+16]
                    with torch.inference_mode():output=model(data.features(ix))
                    for j,i in enumerate(ix):
                        r=rec.loc[i];boxes=mask_boxes(output[j].sigmoid().numpy(),r) if kind=='v31' else decode(output[-1][0][j],output[-1][1][j],r)
                        boxes=[b for b in boxes if b['class_name']==r.class_name and b['confidence']>=.5 and not b['crop_edge']]
                        negative.append(dict(kind=kind,fit_day=fit,seed=seed,id=r.id,frame_id=r.frame_id,class_name=r.class_name,false_positive=bool(boxes),max_confidence=max([b['confidence'] for b in boxes],default=0.)))
    pos=pd.DataFrame(positives);neg=pd.DataFrame(negative);pos.to_csv(A/f'audit/{run}_positive_controls.csv',index=False);neg.to_csv(A/f'audit/{run}_negative_controls.csv',index=False)
    pos['localized']=pos.overlap.ge(.6)
    per_object=pos.groupby(['kind','fit_day','class_name','origin_id','seed']).agg(overlap=('overlap','mean'),coverage=('coverage','mean'),localized=('localized','mean')).reset_index()
    per_object=per_object.groupby(['kind','fit_day','class_name','origin_id'])[['overlap','coverage','localized']].mean().reset_index()
    summary=[]
    for (kind,fit,cls),g in per_object.groupby(['kind','fit_day','class_name']):
        summary.append(dict(kind=kind,fit_day=fit,class_name=cls,objects=len(g),mean_iou=g.overlap.mean(),recall_iou06_per_view=g.localized.mean(),fraction_objects_mean_iou06=g.overlap.ge(.6).mean(),coverage=g.coverage.mean()))
    pd.DataFrame(summary).to_csv(A/f'results/{run}_localization_controls.csv',index=False)
    per_frame=neg.groupby(['kind','fit_day','class_name','frame_id']).false_positive.mean().reset_index();ns=per_frame.groupby(['kind','fit_day','class_name']).agg(frames=('frame_id','size'),false_positive_rate=('false_positive','mean')).reset_index();ns.to_csv(A/f'results/{run}_negative_controls.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False));print(ns.to_string(index=False))

if __name__=='__main__':main()
