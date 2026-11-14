"""Left-seat recall and precision against official held-day annotations."""
from common22 import *
d=pd.read_csv(A/'gpu_results/roi_benchmark.csv')
extra=list((A/'gpu_results/hand_training').glob('*/window_roi_benchmark.csv'))
if extra:d=pd.concat([d]+[pd.read_csv(p) for p in extra],ignore_index=True)
cr=np.array([64,152,400,424]);rows=[]
for (day,variant,scale),g in d.groupby(['day','variant','scale']):
 gt=h[h.day.eq(day)&h.class_name.eq('nude_hands')].copy()
 bb=gt[B].to_numpy();inside=((bb[:,:2]>=cr[:2])&(bb[:,2:]<=cr[2:])).all(1);gt=gt[inside]
 for threshold in [.25,.5,.7]:
  pr=g[g.confidence.ge(threshold)].sort_values('confidence',ascending=False);seen=set();tp=0
  for r in pr.itertuples():
   gg=gt[gt.frame_id.eq(r.frame_id)];ov=iou([getattr(r,c) for c in B],gg[B])[0]
   valid=[(v,ix) for v,ix in zip(ov,gg.index) if v>=.6 and ix not in seen]
   if valid:seen.add(max(valid)[1]);tp+=1
  rows.append(dict(day=day,variant=variant,scale=scale,confidence_threshold=threshold,gt=len(gt),detections=len(pr),true_positive=tp,false_positive=len(pr)-tp,recall=tp/max(len(gt),1),precision=tp/max(len(pr),1),scope='all held-day manual frames; GT fully within fixed left ROI; IoU>=.6'))
pd.DataFrame(rows).to_csv(A/'results/left_seat_detector_benchmark.csv',index=False)
print(pd.DataFrame(rows).query('confidence_threshold==.5').to_string(index=False))
