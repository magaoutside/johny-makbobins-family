"""Independent hand detector + 21-keypoint model, local ONNX CPU inference."""
from common19 import *
sys.path.insert(0,str(A/'vendor'))
from rtmlib import RTMDet,RTMPose
from PIL import Image
import onnxruntime as ort,time
det=RTMDet(str(A/'models/rtmdet_hand.onnx'),model_input_size=(320,320),backend='onnxruntime',device='cpu',score_thr=.4)
pose=RTMPose(str(A/'models/rtmpose_hand.onnx'),model_input_size=(256,256),backend='onnxruntime',device='cpu')
# Avoid excessive thread contention on a shared laptop.
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
for model,name in [(det,'rtmdet_hand'),(pose,'rtmpose_hand')]:model.session=ort.InferenceSession(str(A/f'models/{name}.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
queries=pd.concat([q.assign(kind='query_oof'),te.assign(kind='query_test'),h.assign(kind='manual_train',id=['gt_'+str(i) for i in range(len(h))])],ignore_index=True)
queries=queries[queries.class_name.eq('nude_hands')].copy();cache=A/'rtm_v2_cache';cache.mkdir(exist_ok=True);start=time.time();count=0
for r in queries.itertuples():
    path=cache/f'{r.kind}_{r.id}.json'
    if path.exists():continue
    b=np.array([getattr(r,c) for c in B]);center=(b[:2]+b[2:])/2;wh=b[2:]-b[:2]
    im=Image.open(V16/('test_images' if r.kind=='query_test' else 'images')/f'{r.frame_id}.jpg').convert('RGB');out=[]
    for scale in [2.5,4.]:
        size=max(float(wh.max())*scale,96 if scale==2.5 else 160);roi=np.r_[center-size/2,center+size/2].round().astype(int)
        crop=im.crop(tuple(roi)).resize((384,384),Image.Resampling.BICUBIC)
        for turn in range(4):
            arr=np.ascontiguousarray(np.rot90(np.asarray(crop),turn)[:,:,::-1]);boxes=det(arr)
            if not len(boxes):continue
            keypoints,scores=pose(np.ascontiguousarray(arr[:,:,::-1]),bboxes=boxes)
            for j,(points,score) in enumerate(zip(keypoints,scores)):
                if np.median(score)<.25 or (score>=.15).sum()<16:continue
                xy=points/384
                for _ in range(turn):xy=np.c_[1-xy[:,1],xy[:,0]]
                xy=roi[:2]+xy*(roi[2:]-roi[:2])
                out.append(dict(id=r.id,frame_id=r.frame_id,kind=r.kind,day=str(r.day),scale=scale,turn=turn,hand_index=j,landmarks=xy.tolist(),bbox=np.r_[xy.min(0),xy.max(0)].tolist(),keypoint_scores=score.tolist()))
    path.write_text(json.dumps(out));count+=1
    if count%20==0:print('INFER',count,'/',len(queries),'seconds',round(time.time()-start,1),flush=True)
records=[r for p in cache.glob('*.json') for r in json.loads(p.read_text())];(A/'raw_rtm_v2_landmarks.json').write_text(json.dumps(records))
(A/'rtm_v2_manifest.json').write_text(json.dumps(dict(models={n:hashlib.sha256((A/f'models/{n}.onnx').read_bytes()).hexdigest() for n in ['rtmdet_hand','rtmpose_hand']},source='https://github.com/Tau-J/rtmlib',queries=len(queries),detections=len(records),scales=[2.5,4.],turns=4,minimum_detector_score=.4,minimum_median_keypoint_score=.25,minimum_keypoints_015=16,pose_color_order="RGB per exported pipeline",local_only=True),indent=2))
print('DONE',len(records),'seconds',round(time.time()-start,1),flush=True)
