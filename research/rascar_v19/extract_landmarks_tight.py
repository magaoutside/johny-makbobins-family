"""Local frozen MediaPipe inference. Query and manual-GT evidence kept separate."""
import os
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parent/'cache')
os.environ['GLOG_minloglevel']='2'
from common19 import *
import mediapipe as mp
from PIL import Image
import time
import cv2

options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(A/'models/hand_landmarker.task'),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=4,min_hand_detection_confidence=.35,min_hand_presence_confidence=.5)
queries=pd.concat([q.assign(kind='query_oof'),te.assign(kind='query_test'),h.assign(kind='manual_train',id=['gt_'+str(i) for i in range(len(h))])],ignore_index=True)
queries=queries[queries.class_name.eq('nude_hands')].copy()
cache=A/'landmark_cache_tight';cache.mkdir(exist_ok=True)
start=time.time();count=0
with mp.tasks.vision.HandLandmarker.create_from_options(options) as model:
    for r in queries.itertuples():
        path=cache/f'{r.kind}_{r.id}.json'
        if path.exists():continue
        b=np.array([getattr(r,c) for c in B]);center=(b[:2]+b[2:])/2;wh=b[2:]-b[:2]
        im=Image.open(V16/('test_images' if r.kind=='query_test' else 'images')/f'{r.frame_id}.jpg').convert('RGB');out=[]
        for scale in [1.8,2.8]:
            size=max(float(wh.max())*scale,64 if scale==1.8 else 80)
            roi=np.r_[center-size/2,center+size/2].round().astype(int)
            # Pillow black padding at image borders is explicit and consistent.
            crop=im.crop(tuple(roi)).resize((384,384),Image.Resampling.BICUBIC)
            for turn in range(4):
                rgb=np.asarray(crop)
                if scale==2.8:
                    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB);lab[:,:,0]=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(4,4)).apply(lab[:,:,0]);rgb=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
                arr=np.ascontiguousarray(np.rot90(rgb,turn))
                result=model.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=arr))
                for j,landmarks in enumerate(result.hand_landmarks):
                    xy=np.array([[k.x,k.y] for k in landmarks],float)
                    for _ in range(turn):xy=np.c_[1-xy[:,1],xy[:,0]]
                    xy=roi[:2]+xy*(roi[2:]-roi[:2])
                    rec=dict(id=r.id,frame_id=r.frame_id,kind=r.kind,day=str(r.day),scale=scale,turn=turn,hand_index=j,landmarks=xy.tolist(),bbox=np.r_[xy.min(0),xy.max(0)].tolist())
                    # Handedness confidence is NOT a hand-presence confidence.
                    out.append(rec)
        path.write_text(json.dumps(out));count+=1
        if count%20==0:print('INFER',count,'/',len(queries),'seconds',round(time.time()-start,1),flush=True)
records=[r for p in cache.glob('*.json') for r in json.loads(p.read_text())]
(A/'raw_landmarks_tight.json').write_text(json.dumps(records))
(A/'landmarker_tight_manifest.json').write_text(json.dumps(dict(model='MediaPipe HandLandmarker full float16',model_url='https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',model_sha256=hashlib.sha256((A/'models/hand_landmarker.task').read_bytes()).hexdigest(),mediapipe_version=mp.__version__,queries=len(queries),detections=len(records),scales=[1.8,2.8],min_sizes=[64,80],clahe_scale=2.8,turns=4,minimum_detection_confidence=.35,minimum_presence_confidence=.5,local_only=True),indent=2))
print('DONE',len(queries),len(records),'seconds',time.time()-start,flush=True)
