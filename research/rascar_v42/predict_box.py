"""Research inference API: image + bbox + class -> direct error score.
Coordinates use the dataset's canonical 1280x720 frame. This experimental
classifier did NOT outperform V31 and its sigmoid is not a calibrated probability.
"""
from network import *
import cv2,argparse
class BoxScorer:
 def __init__(self,variant='dual_pool',device=None):
  self.variant=variant;self.device=device or ('mps' if torch.backends.mps.is_available() else 'cpu');base=backbone();self.early=base.features[:6].eval().to(self.device);self.models=[]
  for day in DAYS:
   net=Classifier(base,'tight' if variant=='tight' else 'dual');net.load_state_dict(torch.load(A/f'models/{variant}_fit{day[-2:]}_s4201.pt',map_location='cpu',weights_only=True));self.models.append(net.eval().to(self.device))
 def __call__(self,image_path,box,class_name,neighbours=None,confidence=None):
  assert class_name in CL;b=np.asarray(box,float);assert b.shape==(4,) and np.isfinite(b).all() and np.all(b[2:]>b[:2]) and np.all(b>=0) and np.all(b<=[1280,720,1280,720])
  im=cv2.imread(str(image_path));assert im is not None;im=cv2.resize(im,(1280,720));wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2;span=np.maximum(wh*2,64);lo=np.maximum(0,np.floor(cc-span/2));hi=np.minimum([1280,720],np.ceil(cc+span/2));cb=np.r_[lo,hi];images=[]
  for bb in [b,cb]:
   l=np.floor(bb[:2]).astype(int);r=np.ceil(bb[2:]).astype(int);crop=im[l[1]:r[1],l[0]:r[0],::-1];images.append(cv2.resize(crop,(224,224),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
  rel=(b-np.tile(lo,2))/np.tile(hi-lo,2);nn=np.zeros(5)
  if neighbours is not None and len(neighbours):
   g=pd.DataFrame(neighbours);g=g[g.class_name.eq(class_name)];ov=iou(b,g[B])[0]
   if len(ov):
    other=g.iloc[int(np.argmax(ov))];nn=np.array([max(ov),sum(ov>.5)/5,float(other.confidence),np.log(max((other.x2-other.x1)*(other.y2-other.y1),1)/wh.prod()),float(other.confidence>confidence) if confidence is not None and np.isfinite(confidence) else 0])
  m=np.r_[np.eye(4)[CL.index(class_name)],wh/[1280,720],cc/[1280,720],np.log(wh[0]/wh[1]),rel,nn].astype(np.float32)
  x=torch.tensor(np.stack(images),device=self.device).float()/255;mean=torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=self.device)[None,:,None,None]
  with torch.inference_mode():
   z=self.early((x-mean)/std).unsqueeze(0);mt=torch.tensor(m,device=self.device)[None];scores=[float(net(z,mt).sigmoid()[0]) for net in self.models]
  return dict(score=float(np.mean(scores)),fit25_score=scores[0],fit26_score=scores[1],variant=self.variant,research_only=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--id');p.add_argument('--image');p.add_argument('--box',type=float,nargs=4);p.add_argument('--class-name',choices=CL);p.add_argument('--variant',choices=['tight','dual','dual_pool'],default='dual_pool');p.add_argument('--neighbours-json');args=p.parse_args()
 neighbours=json.loads(Path(args.neighbours_json).read_text()) if args.neighbours_json else [];confidence=None
 if args.id:
  rec=pd.read_csv(A/'inputs/records.csv');r=rec[rec.id.eq(args.id)].iloc[0];args.image=str(D/f'{r.frame_id}.jpg');args.box=r[B].tolist();args.class_name=r.class_name;confidence=r.confidence
  stream=pd.concat([pd.read_csv(D/'train_stream_predictions.csv'),pd.read_csv(D/'test_stream_predictions.csv'),pd.read_csv(D/'machine_predictions.csv')]).drop_duplicates('detection_id');neighbours=stream[stream.frame_id.eq(r.frame_id)&~stream.detection_id.eq(args.id)].to_dict('records')
 assert args.image and args.box and args.class_name,'Provide --id or --image, --box and --class-name'
 print(json.dumps(BoxScorer(args.variant)(args.image,args.box,args.class_name,neighbours,confidence),indent=2))

