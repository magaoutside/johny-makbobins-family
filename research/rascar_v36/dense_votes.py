"""Class-specific dense rectangle voting, not an FCOS/RBF reproduction.

Each class has its own four corner distances at each feature-grid cell.
Within-class overlaps assign the smaller instance. Votes are clustered in
box space before weighted-median decoding. Query geometry never generates
the proposals; it only defines the cached visual crop as in V31.
"""
from common36 import *
import torch,time,cv2,argparse
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(3);cv2.setNumThreads(1)
yy,xx=np.mgrid[:32,:32];xx=(xx+.5)/32;yy=(yy+.5)/32

class VoteHead(nn.Module):
 def __init__(self):
  super().__init__()
  self.trunk=nn.Sequential(nn.Conv2d(384,64,1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU())
  self.cls=nn.Conv2d(64,4,1);self.reg=nn.Conv2d(64,16,1);self.ctr=nn.Conv2d(64,4,1)
  nn.init.constant_(self.cls.bias,-2);nn.init.constant_(self.reg.bias,-1.5)
 def forward(self,x):
  z=self.trunk(F.normalize(x,dim=1)*np.sqrt(384))
  return self.cls(z),F.softplus(self.reg(z).reshape(-1,4,4,32,32)),self.ctr(z)

def targets(r,kind):
 cb=r[CB].to_numpy(float);sz=cb[2:]-cb[:2]
 y=np.zeros((4,32,32),np.float32);dist=np.zeros((4,4,32,32),np.float32);ctr=np.zeros_like(y);weight=np.zeros_like(y)
 assigned=np.full((4,32,32),-1,int)
 for c,cls in enumerate(CL):
  gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(cls)][B].to_numpy(float)
  # Larger objects first; smaller overlapping instances win within a class.
  order=np.argsort(-(gt[:,2:]-gt[:,:2]).prod(1)) if len(gt) else []
  for j in order:
   box=(gt[j]-np.r_[cb[:2],cb[:2]])/np.r_[sz,sz];cx,cy=(box[:2]+box[2:])/2
   l=xx-box[0];t=yy-box[1];rr=box[2]-xx;bb=box[3]-yy
   inside=np.minimum.reduce([l,t,rr,bb])>0
   if kind.startswith('center'):inside&=(abs(xx-cx)<1.5/32)&(abs(yy-cy)<1.5/32)
   y[c,inside]=1
   assigned[c,inside]=-1
   if (box<0).any() or (box>1).any():continue
   assigned[c,inside]=j
   dist[c,:,inside]=np.stack([l,t,rr,bb])[:,inside].T
   ctr[c,inside]=np.sqrt(np.minimum(l,rr)[inside]/np.maximum(l,rr)[inside]*np.minimum(t,bb)[inside]/np.maximum(t,bb)[inside])
  for j in np.unique(assigned[c]):
   if j<0:continue
   mask=assigned[c]==j;weight[c,mask]=1/mask.sum()
 return y,dist,ctr,weight

def loss_fn(pred,target,kind='dense'):
 logits,dist,ctr=pred;y,b,cent,w=target
 if kind=='center_focal':
  p=logits.sigmoid();pt=p*y+(1-p)*(1-y);alpha=.25*y+.75*(1-y)
  cls=(alpha*(1-pt).square()*F.binary_cross_entropy_with_logits(logits,y,reduction='none')).sum()/y.sum().clamp_min(1)
 else:cls=F.binary_cross_entropy_with_logits(logits,y,weight=1+3*y)
 pos=w>0
 if not pos.any():return cls
 a=dist.permute(0,1,3,4,2)[pos];b=b.permute(0,1,3,4,2)[pos];wt=w[pos];wt=wt/wt.sum()
 inter=(torch.minimum(a[:,0],b[:,0])+torch.minimum(a[:,2],b[:,2]))*(torch.minimum(a[:,1],b[:,1])+torch.minimum(a[:,3],b[:,3]))
 aa=(a[:,0]+a[:,2])*(a[:,1]+a[:,3]);bb=(b[:,0]+b[:,2])*(b[:,1]+b[:,3]);ov=inter/(aa+bb-inter).clamp_min(1e-6)
 reg=(wt*(2*(1-ov)+8*F.smooth_l1_loss(a,b,beta=.03,reduction='none').mean(1))).sum()
 cent_loss=(wt*F.binary_cross_entropy_with_logits(ctr[pos],cent[pos],reduction='none')).sum()
 return cls+reg+.3*cent_loss

def wmedian(a,w):
 out=[]
 for col in a.T:
  order=np.argsort(col);cs=np.cumsum(w[order]);out.append(col[order[np.searchsorted(cs,cs[-1]/2)]])
 return np.array(out)

def decode(pred,r,suppression_iou=.45):
 logits,dist,ctr=pred;p=logits.sigmoid().numpy();cent=ctr.sigmoid().numpy();dist=dist.numpy()
 cb=r[CB].to_numpy(float);sz=cb[2:]-cb[:2];rows=[]
 for c,cls in enumerate(CL):
  score=p[c]*(.25+.75*cent[c]);top=np.argsort(score.ravel())[::-1][:400]
  top=top[score.ravel()[top]>=.20]
  if not len(top):continue
  py,px=np.divmod(top,32);l,t,rr,bb=dist[c,:,py,px].T
  boxes=np.stack([xx[py,px]-l,yy[py,px]-t,xx[py,px]+rr,yy[py,px]+bb],1)
  valid=((boxes>=-.02)&(boxes<=1.02)).all(1)&((boxes[:,2:]-boxes[:,:2]).min(1)>.025)
  boxes=np.clip(boxes[valid],0,1);scores=score[py,px][valid];locations=np.stack([xx[py,px],yy[py,px]],1)[valid]
  # Peak decoding is a control, median voting is the main new decoder.
  for mode in ['peak','vote']:
   ids=np.arange(len(boxes));selected=[]
   while len(ids) and len(selected)<8:
    ov=iou(boxes[ids],boxes[ids]);mass=(ov>=.6)@scores[ids]
    seed=int(np.argmax(mass if mode=='vote' else scores[ids]));cluster=ov[seed]>=.6
    ix=ids[cluster];ww=scores[ix]**2
    b=wmedian(boxes[ix],ww) if mode=='vote' else boxes[ids[seed]]
    coherence=float(np.average(iou(b,boxes[ix]).ravel(),weights=ww));support=len(ix)
    # Distinct votes from one image are not independent frames.
    confidence=float(np.average(scores[ix],weights=ww));span=float(np.prod(np.ptp(locations[ix],axis=0)))
    selected.append(b)
    rows.append(dict(class_name=cls,decoder=mode,confidence=confidence,vote_count=support,vote_coherence=coherence,vote_span=span,
                     **dict(zip(B,b*np.r_[sz,sz]+np.r_[cb[:2],cb[:2]]))))
    # Remove all near-duplicates of the selected instance, without chaining.
    ids=ids[iou(b,boxes[ids]).ravel()<suppression_iou]
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=700);ap.add_argument('--kinds',nargs='+',default=['dense','center']);args=ap.parse_args()
 rec=pd.read_csv(V31/'inputs/dense_records.csv');rec['bank']=0
 empty=pd.read_csv(ROOT/'rascar_v33/inputs/empty_records.csv');empty['bank']=1
 assert all(f[-15:-7]==d.replace('-','') for f,d in zip(empty.template_frame,empty.day))
 rec=pd.concat([rec,empty],ignore_index=True)
 feat=[np.load(V31/'features/dense_grids.npy',mmap_mode='r'),np.load(ROOT/'rascar_v33/features/empty_grids.npy',mmap_mode='r')]
 manifest_path=A/'models/training_manifest.json';logs=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
 def features(ids):return torch.from_numpy(np.stack([np.array(feat[int(rec.iloc[i].bank)][int(rec.iloc[i].feature_index)],np.float32) for i in ids]))
 for kind in args.kinds:
  target={i:targets(r,kind) for i,r in rec[rec.split.isin(['manual','empty'])].iterrows()}
  for fit in DAYS:
   torch.manual_seed(3601);rng=np.random.default_rng(3601)
   train=rec[rec.split.isin(['manual','empty'])&rec.day.eq(fit)].index.to_numpy()
   ev=rec[rec.split.eq('test')|(rec.day.ne(fit)&rec.split.isin(['manual','oof','empty']))].index.to_numpy()
   assert not set(rec.loc[train,'frame_id'])&set(rec.loc[ev,'frame_id'])
   groups=[train[rec.loc[train].split.eq(s).to_numpy()] for s in ['manual','empty']];probs=[]
   for group in groups:
    counts=rec.loc[group].groupby('frame_id').size();prob=np.array([1/counts[f] for f in rec.loc[group,'frame_id']]);probs.append(prob/prob.sum())
   name=f'{kind}_fit{fit[-2:]}';model=VoteHead();opt=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.02);running=[];start=time.time()
   for step in range(args.steps):
    ids=np.concatenate([rng.choice(g,n,replace=True,p=p) for g,n,p in zip(groups,[6,2],probs)]);x=features(ids)
    tt=tuple(torch.from_numpy(np.stack([target[i][k] for i in ids])) for k in range(4))
    loss=loss_fn(model(x),tt,kind);assert torch.isfinite(loss)
    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();running.append(float(loss.detach()))
    if step%100==0:print('FIT',name,step,round(np.mean(running[-100:]),4),round(time.time()-start,1),flush=True)
   torch.save(model.state_dict(),A/f'models/{name}.pt');model.eval();out=[]
   for off in range(0,len(ev),16):
    ids=ev[off:off+16]
    with torch.inference_mode():pred=model(features(ids))
    for j,i in enumerate(ids):
     r=rec.loc[i]
     for box in decode(tuple(v[j] for v in pred),r):out.append(dict(record_id=r.id,record_split=r.split,frame_id=r.frame_id,fit_day=fit,**box))
   pd.DataFrame(out).to_csv(A/f'features/{name}_boxes.csv',index=False)
   logs=[m for m in logs if m['name']!=name]
   logs.append(dict(name=name,fit_day=fit,train_crops=len(train),fit_frames=sorted(rec.loc[train,'frame_id'].unique()),split_counts=rec.loc[train].split.value_counts().to_dict(),steps=args.steps,seed=3601,last_loss=float(np.mean(running[-100:])),seconds=time.time()-start,test_labels_used=False,backbone='facebook/dinov2-small',regression='class-specific LTRB',supervision=kind))
   (A/'models/training_manifest.json').write_text(json.dumps(logs,indent=2));print('DONE',name,len(out),round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
