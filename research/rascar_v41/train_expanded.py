"""Equal-step clean/pool controls and a separate automatic test adaptation.
Test adaptation has NO claimed OOF metric. No human test decisions enter here.
"""
from vision import *
import argparse,time,gc
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('key',choices=['vit_b','vit_l']);p.add_argument('--transductive',action='store_true');p.add_argument('--fold-adapt',action='store_true');p.add_argument('--steps',type=int,default=250);p.add_argument('--seed',type=int,default=4142);p.add_argument('--pool-source',choices=['all','old','new'],default='all');p.add_argument('--expanded-only',action='store_true');args=p.parse_args();key=args.key
assert not (args.transductive and args.fold_adapt)
tag='' if args.seed==4142 and args.pool_source=='all' else f'_s{args.seed}_{args.pool_source}'
device='mps';assert torch.backends.mps.is_available();P=A/'pool_expansion';q,te,h=inputs();base=pd.read_csv(A/'inputs/records.csv');rec=base.copy();features=np.load(A/f'features/{key}_final_patches.npy',mmap_mode='r');extra_features=np.load(P/f'{key}_patches.npy',mmap_mode='r');candidates=pd.read_csv(P/'candidates.csv');review=pd.read_csv(P/'approved_geometry.csv');review_ids=set(review.id);decisions=review.set_index('id').decision.to_dict()
extra=candidates[candidates.id.isin(review_ids)].copy();extra['feature_source']='pool';extra['feature_index']=extra.index;extra['split']='reviewed_geometry';extra['origin_id']=extra.id;extra['id']='new_'+extra.id;extra['negative']=extra.origin_id.map(decisions).eq('class_absent');extra['weight']=.25
rec['feature_source']='base';rec['feature_index']=np.arange(len(rec));rec['negative']=False
old=pd.read_csv(ROOT/'rascar_v33/inputs/pseudo_records.csv');old_ids=set(old.loc[old.label_use.eq('geometry_and_presence'),'id']);old_mask=rec.id.isin(old_ids);rec.loc[old_mask,'split']='old_geometry';rec.loc[old_mask,'weight']=.15
rec=pd.concat([rec,extra],ignore_index=True)
if args.transductive:
 evidence=pd.read_csv(P/'evidence.csv');_,base_test=baseline(q,te);scores=dict(zip(te.id,base_test))
 auto=evidence[evidence.split.eq('auto_test')&evidence.geometry_admit&evidence.confidence.ge(.8)].copy();auto=auto[auto.id.map(scores).lt(.4)]
 auto_ids=set(auto.id);auto=candidates[candidates.id.isin(auto_ids)&candidates.split.eq('auto_test')].copy();auto['feature_source']='pool';auto['feature_index']=auto.index;auto['origin_id']=auto.id;auto['id']='auto_'+auto.id;auto['split']='auto_geometry';auto['negative']=False;auto['weight']=.15;rec=pd.concat([rec,auto],ignore_index=True)
else:auto=pd.DataFrame()
if args.fold_adapt:
 # Selection sees only predictions from the OTHER day. Human held-day boxes
 # and error labels are not consulted. Pseudo targets are the source query box.
 bo,_=baseline(q,te);eligible=q.set_index('id').confidence.ge(.8)
 for teacher in ['vit_b','vit_l']:
  z=pd.read_csv(A/f'features/{teacher}_geometry_predictions.csv');z=z[z.split.eq('query')].set_index('id').loc[q.id]
  assert (z.fit_day.to_numpy()!=q.day.to_numpy()).all()
  eligible&=z.reference_iou.ge(.68)&z.reference_confidence.ge(.7)
 z=pd.concat([pd.read_csv(A/f'features/yolo_fit{v}_predictions.csv') for v in ['25','26']]);z=z[z.split.eq('query')].set_index('id').loc[q.id]
 assert (z.fit_day.to_numpy()!=q.day.to_numpy()).all()
 eligible&=z.assigned_iou.ge(.6)&z.reference_confidence.ge(.35)&pd.Series(bo<.4,index=q.id)
 auto=rec[rec.split.eq('query')&rec.id.isin(eligible[eligible].index)].copy();auto['origin_id']=auto.id;auto['id']='fold_auto_'+auto.id;auto['split']='auto_geometry';auto['negative']=False;auto['weight']=.15;rec=pd.concat([rec,auto],ignore_index=True)
grid=16;dim=features.shape[1];mask,extra_tensor=aux(rec,grid);targets=np.zeros((len(rec),1,grid,grid),np.float32);valid=np.zeros(len(rec),bool)
for ix,r in rec.iterrows():
 if r.split in ['old_geometry','reviewed_geometry','auto_geometry']:
  valid[ix]=True;box=None if r.negative else r[B].to_numpy(float)
 elif r.split in ['query','synthetic']:
  valid[ix]=True;gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy();ov=iou(r[B],gt)[0] if len(gt) else [];box=gt[np.argmax(ov)] if len(ov) and max(ov)>=.05 else None
 else:continue
 if box is None:continue
 cb=r[CB].to_numpy(float);size=cb[2:]-cb[:2];b=(box-np.tile(cb[:2],2))/np.tile(size,2);lo=np.arange(grid)/grid;hi=(np.arange(grid)+1)/grid
 wx=np.maximum(0,np.minimum(b[2],hi)-np.maximum(b[0],lo))*grid;wy=np.maximum(0,np.minimum(b[3],hi)-np.maximum(b[1],lo))*grid;targets[ix,0]=wy[:,None]*wx[None,:]
class Geometry(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(dim+7,96,1),nn.GroupNorm(8,96),nn.GELU(),nn.Conv2d(96,48,3,padding=1),nn.GroupNorm(8,48),nn.GELU(),nn.Conv2d(48,1,1))
 def forward(self,x,m,e):
  n=len(x);xy=torch.stack(torch.meshgrid(torch.linspace(-1,1,16,device=x.device),torch.linspace(-1,1,16,device=x.device),indexing='ij')).expand(n,-1,-1,-1);cls=e[:,-4:,None,None].expand(-1,-1,16,16)
  return self.net(torch.cat([F.normalize(x,dim=1)*np.sqrt(dim),m.reshape(n,1,16,16),xy,cls],1))
def batch(ii):
 return torch.tensor(np.stack([features[int(rec.iloc[i].feature_index)] if rec.iloc[i].feature_source=='base' else extra_features[int(rec.iloc[i].feature_index)] for i in ii]).astype(np.float32),device=device)
outputs=[];manifest=[]
settings=[(fit,variant) for fit in DAYS for variant in (['expanded_pool'] if args.expanded_only else ['clean_continue','expanded_pool'])] if not args.transductive else [('all','final_no_test'),('all','final_auto_test')]
if args.fold_adapt:settings=[(fit,variant) for fit in DAYS for variant in ['fold_control','fold_auto']]
for fit,variant in settings:
 core=rec.split.isin(['query','synthetic'])&valid;pool=rec.split.isin(['old_geometry','reviewed_geometry'])&valid
 if args.pool_source=='old':pool&=rec.split.eq('old_geometry')
 if args.pool_source=='new':pool&=rec.split.eq('reviewed_geometry')
 if fit!='all':core&=rec.day.eq(fit);pool&=rec.day.eq(fit)
 use=core.copy()
 if variant!='clean_continue':use|=pool
 if variant=='final_auto_test':use|=rec.split.eq('auto_geometry')&valid
 if variant=='fold_auto':use|=rec.split.eq('auto_geometry')&rec.day.ne(fit)&valid
 ev=rec.split.eq('test')|((rec.split.eq('query')&rec.day.ne(fit)) if fit!='all' else False);ix=np.flatnonzero(use);ex=np.flatnonzero(ev)
 if variant not in ['final_auto_test','fold_auto']:assert not set(rec.loc[use,'frame_id'])&set(rec.loc[ev,'frame_id'])
 assert not (use&rec.split.isin(['query','synthetic'])&rec.day.ne(fit)).any() if fit!='all' else True
 torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed);net=Geometry().to(device);net.load_state_dict(torch.load(A/f'models/{key}_geometry_fit{fit[-2:] if fit!="all" else "25"}.pt',map_location=device,weights_only=True));opt=torch.optim.AdamW(net.parameters(),lr=.00025,weight_decay=.02)
 prob=np.zeros(len(rec));
 for group,budget in [(core,1. if variant=='clean_continue' else .75),(pool,.25)]:
  g=group&use
  if g.any():
   sub=rec.loc[g].copy()
   if group is pool:
    times=pd.to_datetime(sub.frame_id.str.split('_').str[-2:].str.join(''),format='%Y%m%d%H%M%S').astype('int64')//600000000000
    sub['cluster']=sub.class_name+'_'+times.astype(str)+'_'+(((sub.x1+sub.x2)/2)//80).astype(int).astype(str)+'_'+(((sub.y1+sub.y2)/2)//80).astype(int).astype(str)
    counts=sub.groupby('cluster').id.transform('size').to_numpy()
   else:counts=sub.groupby('frame_id').id.transform('size').to_numpy()
   w=1/counts;prob[g]=budget*w/w.sum()
 if variant in ['final_auto_test','fold_auto']:
  g=use&rec.split.eq('auto_geometry')
  if g.any():prob[g]=.15/g.sum()
 prob=prob[ix]/prob[ix].sum();start=time.time()
 for step in range(args.steps):
  ii=rng.choice(ix,32,replace=True,p=prob);x=batch(ii);m=mask[ii].clone();ext=extra_tensor[ii].clone()
  # Pseudo targets are kept fixed; query masks are jittered to prevent copying.
  selected=rec.iloc[ii].split.isin(['old_geometry','reviewed_geometry','auto_geometry']).to_numpy()
  if selected.any():
   sub=rec.iloc[ii[selected]].copy();b=sub[B].to_numpy();wh=b[:,2:]-b[:,:2];cc=(b[:,:2]+b[:,2:])/2;shift=rng.uniform(-.15,.15,wh.shape)*wh;size=wh*rng.uniform(.7,1.3,(len(sub),1));sub[B]=np.c_[cc+shift-size/2,cc+shift+size/2];mm,ee=aux(sub,16);m[selected]=mm;ext[selected]=ee
  y=torch.tensor(targets[ii],device=device);logits=net(x,m.to(device),ext.to(device));pr=logits.sigmoid();bce=F.binary_cross_entropy_with_logits(logits,y,reduction='none').mean((1,2,3));dice=1-(2*(pr*y).sum((1,2,3))+1)/(pr.sum((1,2,3))+y.sum((1,2,3))+1)
  w=np.where(selected,rec.iloc[ii].weight.to_numpy(),1.);loss=((bce+.3*dice)*torch.tensor(w,device=device,dtype=torch.float32)).mean();opt.zero_grad();loss.backward();opt.step()
  if step%100==0:print('EXPANDED',key,fit,variant,step,float(loss),round(time.time()-start,1),flush=True)
 net.eval()
 with torch.inference_mode():
  for off in range(0,len(ex),24):
   ii=ex[off:off+24];pr=net(batch(ii),mask[ii].to(device),extra_tensor[ii].to(device)).sigmoid().cpu().numpy()[:,0]
   for ixr,pm in zip(ii,pr):
    r=rec.iloc[ixr];cb=r[CB].to_numpy(float);size=cb[2:]-cb[:2];n,lab,stats,cent=cv2.connectedComponentsWithStats((pm>=.5).astype(np.uint8),8);boxes=[]
    for j in range(1,n):
     x,y,w,hh,ar=stats[j]
     if ar>=2:boxes.append(np.array([x,y,x+w,y+hh])/16*np.tile(size,2)+np.tile(cb[:2],2))
    ovs=iou(r[B],boxes)[0] if boxes else [];ov=float(max(ovs)) if len(ovs) else 0.;bb=boxes[int(np.argmax(ovs))] if len(ovs) else [np.nan]*4
    outputs.append(dict(id=r.id,frame_id=r.frame_id,day=r.day,class_name=r.class_name,split=r.split,fit_day=fit,variant=variant,key=key,score=float(np.clip((.85-ov)/.5,0,1)),reference_iou=ov,**dict(zip(['ref_'+b for b in B],bb))))
 torch.save({k:v.cpu() for k,v in net.state_dict().items()},A/f'models/{key}_{variant}_fit{fit[-2:]}{tag}.pt')
 manifest.append(dict(key=key,fit_day=fit,variant=variant,steps=args.steps,seed=args.seed,pool_source=args.pool_source,train_frames=sorted(rec.loc[use,'frame_id'].unique()),counts=rec.loc[use].split.value_counts().to_dict(),automatic_test_ids=rec.loc[use&rec.split.eq('auto_geometry'),'origin_id'].tolist() if 'origin_id' in rec else [],manual_test_labels_used=False,claims_oof=fit!='all',validation_protocol='transductive held-day images without labels' if args.fold_adapt else 'inductive cross-day' if fit!='all' else 'no validation metric',seconds=time.time()-start))
 del net,opt;gc.collect();torch.mps.empty_cache()
suffix='foldadapt' if args.fold_adapt else 'transductive' if args.transductive else 'expanded';pd.DataFrame(outputs).to_csv(A/f'features/{key}_{suffix}{tag}_predictions.csv',index=False);(A/f'models/{key}_{suffix}{tag}_manifest.json').write_text(json.dumps(manifest,indent=2))
print('DONE',key,suffix,flush=True)
