from pathlib import Path
import numpy as np,pandas as pd,cv2,json,time,torch
from rtmlib import RTMPose,YOLOX
A=Path('/kaggle/working/rascar_v25');B=['x1','y1','x2','y2'];paths=json.loads((A/'image_paths.json').read_text());q=pd.read_csv(A/'queries.csv');people=pd.read_csv(A/'persons.csv');cv2.setNumThreads(2);torch.set_num_threads(2)
url='https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip'
model=RTMPose(url,model_input_size=(288,384),backend='onnxruntime',device='cuda');det=YOLOX('/root/.cache/rtmlib/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx',model_input_size=(640,640),backend='onnxruntime',device='cuda')
start=time.time();frames=sorted(q.frame_id.unique());rec={m:[] for m in ['rtmwx_bgr','rtmwx_rgb']};poses={m:[] for m in rec}
for ii,fid in enumerate(frames):
 im=cv2.resize(cv2.imread(paths[fid]),(1280,720));boxes=list(det(im));extra=people[people.frame_id.eq(fid)][B].to_numpy()
 for bb in extra:
  def ov(b,c):
   i=np.maximum(0,np.minimum(b[2:],c[2:])-np.maximum(b[:2],c[:2])).prod();return i/max((b[2:]-b[:2]).prod()+(c[2:]-c[:2]).prod()-i,1)
  if all(ov(bb,b)<.6 for b in boxes):boxes.append(bb)
 if not boxes:continue
 for name in rec:
  pic=im if name.endswith('bgr') else im[:,:,::-1].copy();kp,sc=model(pic,bboxes=np.array(boxes))
  for pi,(xy,co) in enumerate(zip(kp,sc)):
   for side,sl in [('left',slice(91,112)),('right',slice(112,133))]:
    pts=xy[sl];cf=co[sl];good=cf>=max(.2,float(np.max(cf))*.15)
    poses[name].append(dict(frame_id=fid,turn=0,person=pi,side=side,xy=json.dumps(pts.tolist()),confidence=json.dumps(cf.tolist())))
    if good.sum()<8:continue
    b=np.r_[pts[good].min(0),pts[good].max(0)]
    if min(b[2:]-b[:2])<2:continue
    rec[name].append(dict(frame_id=fid,query_id='',variant=name,turn=0,side=side,confidence=float(np.median(cf)),points_above02=int(good.sum()),**dict(zip(B,b))))
 if ii%20==0:print('VARIANT_PROGRESS',ii,flush=True)
for name,rows in rec.items():
 out=A/name;out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'references.csv',index=False);pd.DataFrame(poses[name]).to_csv(out/'keypoints.csv',index=False);(out/'DONE.json').write_text(json.dumps(dict(model=name,url=url,frames=len(frames),references=len(rows),seconds=time.time()-start,raw_simcc_scores_not_probabilities=True)))
print('VARIANTS_DONE',flush=True)
