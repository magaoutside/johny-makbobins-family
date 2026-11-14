"""Peak decoding and duplicate-center fusion before watershed separation.
No additional model fitting, no test annotation input, identical folds to V31.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent;V31=ROOT/'rascar_v31';A=ROOT/'rascar_v32'
sys.path.insert(0,str(V31))
from dense_train import Head,torch,np,pd,F,CL,B,DAYS,iou
from watershed_boxes import separate
import time

def peak_decode(pred,r):
 logits,dist,ctr=pred;score=torch.sqrt(logits.sigmoid()*ctr.sigmoid());peak=F.max_pool2d(score[None],3,1,1)[0];prob=score.numpy();valid=(score>=peak).numpy();dd=dist.numpy()
 cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(float);wh=cb[2:]-cb[:2];out=[]
 for k,cls in enumerate(CL):
  candidates=[]
  for idx in np.argsort(prob[k].ravel())[::-1]:
   y,x=divmod(int(idx),32);conf=float(prob[k,y,x])
   if conf<.08:break
   if not valid[k,y,x]:continue
   l,t,rr,bb=dd[:,y,x];cx=(x+.5)/32;cy=(y+.5)/32
   box=np.clip([cx-l,cy-t,cx+rr,cy+bb],0,1)*np.r_[wh,wh]+np.r_[cb[:2],cb[:2]]
   if candidates and iou(box,np.array([z['box'] for z in candidates])).max()>.5:continue
   candidates.append(dict(box=box,confidence=conf))
   if len(candidates)>=12:break
  for z in candidates:out.append(dict(class_name=cls,threshold=0.,confidence=z['confidence'],**dict(zip(B,z['box']))))
 return pd.DataFrame(out,columns=['class_name','threshold','confidence']+B)

def fuse_centers(d):
 """Greedy clusters around strongest reference, no transitive chaining.
 Broad, low-overlap proposals do not connect two otherwise distinct hands.
 """
 rows=[]
 for cls,g in d.groupby('class_name'):
  g=g.sort_values('confidence',ascending=False).copy()
  while len(g):
   best=g.iloc[0];bb=best[B].to_numpy(float);boxes=g[B].to_numpy(float);center=(boxes[:,:2]+boxes[:,2:])/2;bc=(bb[:2]+bb[2:])/2
   sizes=np.maximum(boxes[:,2:]-boxes[:,:2],1);bs=np.maximum(bb[2:]-bb[:2],1)
   distance=np.linalg.norm((center-bc)/np.minimum(sizes,bs),axis=1);ov=iou(bb,boxes)[0]
   group=(ov>=.25)&(distance<=.55);group[0]=True
   weights=g.loc[group,'confidence'].to_numpy()**2;box=np.average(boxes[group],axis=0,weights=weights)
   rows.append(dict(class_name=cls,threshold=0.,confidence=float(best.confidence),fused_count=int(group.sum()),**dict(zip(B,box))))
   g=g.loc[~group]
 return pd.DataFrame(rows,columns=['class_name','threshold','confidence','fused_count']+B)

def main():
 rec=pd.read_csv(A/'inputs/dense_records.csv');feat=np.load(V31/'features/dense_grids.npy',mmap_mode='r');log=[]
 for fit in DAYS:
  det=Head('conv_detector');det.load_state_dict(torch.load(V31/f'models/conv_detector_fit{fit[-2:]}.pt',weights_only=True));det.eval()
  seg=Head('linear_mask');seg.load_state_dict(torch.load(V31/f'models/linear_mask_fit{fit[-2:]}.pt',weights_only=True));seg.eval()
  ev=rec[rec.split.eq('test')|rec.day.ne(fit)].index.to_numpy();outputs={k:[] for k in ['peak_detector','fused_detector','watershed_peak','watershed_fused']};start=time.time()
  for off in range(0,len(ev),16):
   ids=ev[off:off+16];x=torch.from_numpy(np.array(feat[ids],dtype=np.float32))
   with torch.inference_mode():pr=det(x);mask=seg(x).sigmoid().numpy()
   for j,i in enumerate(ids):
    r=rec.loc[i];raw=peak_decode(tuple(z[j] for z in pr),r);fused=fuse_centers(raw)
    meta=dict(record_id=r.id,record_split=r.split,frame_id=r.frame_id,fit_day=fit)
    for key,d in [('peak_detector',raw),('fused_detector',fused)]:outputs[key].extend([dict(**meta,**row) for row in d.to_dict('records')])
    for key,d in [('watershed_peak',raw),('watershed_fused',fused)]:outputs[key].extend([dict(**meta,**box) for box in separate(mask[j],d,r)])
   if off%160==0:print('REFINE',fit,off,len(ev),round(time.time()-start,1),flush=True)
  for key,rows in outputs.items():pd.DataFrame(rows).to_csv(A/f'features/{key}_fit{fit[-2:]}_boxes.csv',index=False)
  print('DONE',fit,{k:len(v) for k,v in outputs.items()},round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
