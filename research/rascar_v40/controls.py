"""Same held-day positive/empty crop diagnostics for the resolution ablation."""
from common40 import *
from set_detector import Data,SEEDS
import argparse

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',default=['refine16','refine32']);args=p.parse_args();data=Data();rec=data.rec;positive=[];negative=[]
    for run in args.runs:
        for fit in DAYS:
            source=rec[rec.day.ne(fit)&rec.split.isin(['manual','empty'])&rec.class_name.isin(CL)]
            for seed in SEEDS:
                pred=pd.read_csv(A/f'features/{run}_fit{fit[-2:]}_seed{seed}_boxes.csv');pred=pred[pred.confidence.ge(.5)&~pred.crop_edge]
                groups={k:g for k,g in pred.groupby(['record_id','class_name'])}
                for _,r in source.iterrows():
                    boxes=groups.get((r.id,r.class_name),pd.DataFrame(columns=B))[B].to_numpy(float)
                    if r.split=='manual':
                        v=float(iou(r[B],boxes).max()) if len(boxes) else 0.
                        positive.append(dict(run=run,fit_day=fit,seed=seed,id=r.id,origin_id=r.origin_id,frame_id=r.frame_id,class_name=r.class_name,overlap=v,localized=v>=.6,coverage=bool(len(boxes))))
                    else:negative.append(dict(run=run,fit_day=fit,seed=seed,id=r.id,frame_id=r.frame_id,class_name=r.class_name,false_positive=bool(len(boxes))))
    pos=pd.DataFrame(positive);neg=pd.DataFrame(negative);pos.to_csv(A/'audit/resolution_positive_controls.csv',index=False);neg.to_csv(A/'audit/resolution_negative_controls.csv',index=False)
    per_object=pos.groupby(['run','fit_day','class_name','origin_id'])[['overlap','localized','coverage']].mean().reset_index();out=[]
    for (run,fit,cls),g in per_object.groupby(['run','fit_day','class_name']):out.append(dict(run=run,fit_day=fit,class_name=cls,objects=len(g),mean_iou=g.overlap.mean(),recall_iou06_per_view=g.localized.mean(),coverage=g.coverage.mean()))
    base=pd.read_csv(ROOT/'rascar_v39/results/set_long_localization_controls.csv').rename(columns={'kind':'run'});summary=pd.concat([base,pd.DataFrame(out)],ignore_index=True);summary.to_csv(A/'results/localization_controls.csv',index=False)
    frames=neg.groupby(['run','fit_day','class_name','frame_id']).false_positive.mean().reset_index();ns=frames.groupby(['run','fit_day','class_name']).agg(frames=('frame_id','size'),false_positive_rate=('false_positive','mean')).reset_index();ns.to_csv(A/'results/negative_controls.csv',index=False)
    print(summary[summary.class_name.eq('nude_hands')].to_string(index=False))

if __name__=='__main__':main()
