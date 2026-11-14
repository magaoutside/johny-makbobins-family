"""Re-extract visual features at two new context scales, then infer frozen heads.
The query is not resized or changed; only the surrounding visual context changes.
"""
from dense_votes import *
from transformers import AutoModel
import os
os.environ['TOKENIZERS_PARALLELISM']='false'

def prepare():
 base=pd.read_csv(V31/'inputs/dense_records.csv');base=base[base.split.isin(['oof','test'])&base.class_name.eq('person')]
 rows=[]
 for _,r in base.iterrows():
  b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2
  for scale in [1.8,3.2]:
   cb=np.r_[np.maximum(0,np.floor(cc-scale*wh/2)),np.minimum([1280,720],np.ceil(cc+scale*wh/2))]
   d=r.to_dict();d.update(dict(zip(CB,cb)));d.update(view_scale=scale,feature_index=len(rows));rows.append(d)
 rec=pd.DataFrame(rows);rec.to_csv(A/'inputs/multiscale_records.csv',index=False);return rec

def extract(rec):
 path=A/'features/multiscale_grids.npy'
 if (A/'features/multiscale_complete.json').exists():
  z=np.load(path,mmap_mode='r');assert len(z)==len(rec);return z
 model=AutoModel.from_pretrained(str(V31/'model_cache/dinov2-small'),local_files_only=True,trust_remote_code=False).eval()
 dst=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(len(rec),384,32,32))
 mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1);start=time.time()
 for off in range(0,len(rec),4):
  batch=[]
  for _,r in rec.iloc[off:off+4].iterrows():
   im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));b=r[CB].to_numpy(int);im=im[b[1]:b[3],b[0]:b[2],::-1]
   batch.append(cv2.resize(im,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
  x=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std
  with torch.inference_mode():z=model(pixel_values=x).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,32,32)
  dst[off:off+len(batch)]=z.numpy().astype(np.float16)
  if off%80==0:print('MULTISCALE',off,len(rec),round(time.time()-start,1),flush=True)
 dst.flush();(A/'features/multiscale_complete.json').write_text(json.dumps(dict(count=len(rec),scales=[1.8,3.2],seconds=time.time()-start,backbone='facebook/dinov2-small',test_labels_used=False),indent=2));return np.load(path,mmap_mode='r')

def linear_decode(prob,r):
 k=CL.index(r.class_name);p=prob[k];n,lab,stats,_=cv2.connectedComponentsWithStats((p>=.5).astype(np.uint8),8);out=[]
 cb=r[CB].to_numpy(float);sz=cb[2:]-cb[:2]
 for j in range(1,n):
  x,y,w,hh,ar=stats[j]
  if ar<4:continue
  b=np.r_[x,y,x+w,y+hh]/32*np.r_[sz,sz]+np.r_[cb[:2],cb[:2]]
  out.append(dict(class_name=r.class_name,decoder='linear',confidence=float(p[lab==j].mean()),vote_count=ar,vote_coherence=1.,vote_span=0.,**dict(zip(B,b))))
 return out

def main():
 rec=prepare();feat=extract(rec)
 for fit in DAYS:
  vote=VoteHead();vote.load_state_dict(torch.load(A/f'models/dense_fit{fit[-2:]}.pt',weights_only=True));vote.eval()
  linear=nn.Conv2d(384,4,1);state=torch.load(V31/f'models/linear_mask_fit{fit[-2:]}.pt',weights_only=True);linear.load_state_dict({k.replace('net.',''):v for k,v in state.items()});linear.eval()
  ev=rec[rec.split.eq('test')|rec.day.ne(fit)].index.to_numpy();out=[]
  for off in range(0,len(ev),16):
   ids=ev[off:off+16];x=torch.from_numpy(np.array(feat[ids],np.float32))
   with torch.inference_mode():vp=vote(x);lp=linear(F.normalize(x,dim=1)*np.sqrt(384)).sigmoid().numpy()
   for j,i in enumerate(ids):
    r=rec.loc[i]
    for b in decode(tuple(v[j] for v in vp),r)+linear_decode(lp[j],r):out.append(dict(record_id=r.id,record_split=r.split,frame_id=r.frame_id,fit_day=fit,view_scale=r.view_scale,**b))
  # Existing 2.4x predictions are the third view, reused without recomputation.
  current=pd.read_csv(A/f'features/dense_fit{fit[-2:]}_boxes.csv');current=current[current.record_split.isin(['oof','test'])&current.class_name.eq('person')].copy();current['view_scale']=2.4
  old=pd.read_csv(V31/f'features/linear_mask_fit{fit[-2:]}_boxes.csv');old=old[old.record_split.isin(['oof','test'])&old.class_name.eq('person')&old.threshold.eq(.5)].copy();old['view_scale']=2.4;old['decoder']='linear';old['vote_count']=0;old['vote_coherence']=1.;old['vote_span']=0.
  pd.concat([pd.DataFrame(out),current,old],ignore_index=True).to_csv(A/f'features/multiscale_fit{fit[-2:]}_boxes.csv',index=False)
  print('MULTISCALE_INFER',fit,len(out),flush=True)

if __name__=='__main__':main()
