"""Frozen independent DeepLabV3 hand segmentation, two crop contexts.
No RASCAR fitting. Official author's weights, ImageNet normalization.
Missing segmentation is not converted into an error label.
"""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v32';sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,h,B,iou
from features23 import image,paths
import numpy as np,pandas as pd,cv2,torch
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from safetensors.torch import load_file
torch.set_num_threads(3);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
TAG='handseg_wide' if '--wide' in sys.argv else 'handseg'
CONTEXTS=[8.,12.] if '--wide' in sys.argv else [2.4,4.]

def main():
 paths.update({p.stem:p for p in (ROOT/'rascar_v25/gpu/results/pool_images').glob('*.jpg')})
 manual=h[h.class_name.eq('nude_hands')].copy();manual['id']=['manual_gt_'+str(i) for i in manual.index]
 pool=pd.read_csv(ROOT/'rascar_v22/inputs/reviewed_pool_hands.csv')
 records=pd.concat([q[q.class_name.eq('nude_hands')].assign(split='oof'),te[te.class_name.eq('nude_hands')].assign(split='test'),manual.assign(split='manual'),pool.assign(split='pool')],ignore_index=True)
 tasks=[]
 for _,r in records.iterrows():
  b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);center=(b[:2]+b[2:])/2
  for context in CONTEXTS:
   cb=np.r_[np.maximum(0,np.floor(center-context*wh/2)),np.minimum([1280,720],np.ceil(center+context*wh/2))].astype(int)
   tasks.append(dict(id=r.id,frame_id=r.frame_id,split=r.split,context=context,**dict(zip(B,b)),**dict(zip(['crop_x1','crop_y1','crop_x2','crop_y2'],cb))))
 tasks=pd.DataFrame(tasks);tasks.to_csv(A/f'inputs/{TAG}_records.csv',index=False)
 model=deeplabv3_resnet50(weights=None,weights_backbone=None,num_classes=2,aux_loss=False);model.load_state_dict(load_file(str(A/'model_cache/handseg/weights.safetensors')),strict=True);model.eval()
 mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
 maps=np.lib.format.open_memmap(A/f'features/{TAG}_maps.npy',mode='w+',dtype=np.float16,shape=(len(tasks),256,256));summary=[];boxes=[];start=time.time()
 for off in range(0,len(tasks),4):
  batch=[]
  for _,r in tasks.iloc[off:off+4].iterrows():
   cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(int);im=image(r.frame_id)[cb[1]:cb[3],cb[0]:cb[2],::-1]
   # Match torchvision's PIL bilinear resize used by the author's inference.
   im=np.asarray(Image.fromarray(im).resize((256,256),Image.Resampling.BILINEAR));batch.append(im.transpose(2,0,1).copy())
  x=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std
  with torch.inference_mode():prob=model(x)['out'].softmax(1)[:,1].numpy()
  maps[off:off+len(prob)]=prob.astype(np.float16)
  for (idx,r),p in zip(tasks.iloc[off:off+len(prob)].iterrows(),prob):
   cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(float);wh=cb[2:]-cb[:2];b=r[B].to_numpy(float);z=(b-np.r_[cb[:2],cb[:2]])/np.r_[wh,wh]*256;lo=np.clip(np.floor(z[:2]).astype(int),0,255);hi=np.clip(np.ceil(z[2:]).astype(int),1,256);inside=p[lo[1]:hi[1],lo[0]:hi[0]]
   summary.append(dict(id=r.id,split=r.split,context=r.context,prob_mean=float(inside.mean()),prob_p95=float(np.quantile(inside,.95)),fraction_hand=float((inside>=.5).mean())))
   for cut in [.35,.5,.65]:
    n,lab,stats,cents=cv2.connectedComponentsWithStats((p>=cut).astype(np.uint8),8)
    for j in range(1,n):
     x,y,w,hh,area=stats[j]
     if area<64:continue
     bb=np.r_[x,y,x+w,y+hh]/256*np.r_[wh,wh]+np.r_[cb[:2],cb[:2]]
     boxes.append(dict(id=r.id,split=r.split,context=r.context,threshold=cut,confidence=float(p[lab==j].mean()),touch_crop=bool(x==0 or y==0 or x+w==256 or y+hh==256),area_pixels=int(area),**dict(zip(B,bb))))
  if off%80==0:print('HANDSEG',off,len(tasks),round(time.time()-start,1),flush=True)
 maps.flush();pd.DataFrame(summary).to_csv(A/f'features/{TAG}_summary.csv',index=False);pd.DataFrame(boxes).to_csv(A/f'features/{TAG}_boxes.csv',index=False)
 (A/f'features/{TAG}_manifest.json').write_text(json.dumps(dict(records=len(records),views=len(tasks),seconds=time.time()-start,pretrained='guglielmocamporese/hands-segmentation-pytorch',rascar_fitting=False,contexts=CONTEXTS,model='DeepLabV3 ResNet50',input=[256,256]),indent=2));print('DONE',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
