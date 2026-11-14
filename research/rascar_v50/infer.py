"""Independent pretrained OWLv2 local references and semantic contrasts."""
from pathlib import Path
import sys,json,time,argparse,os
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v41'))
from common import inputs,iou,B,CL,D
sys.path.pop(0)
import numpy as np,pandas as pd,torch,cv2
from transformers import Owlv2Processor,Owlv2ForObjectDetection
from PIL import Image
from functools import lru_cache
PROMPTS={
 'person':(['a photo of a person','a photo of a seated person'],['a photo of a chair','a photo of a table']),
 'nude_hands':(['a photo of a human hand','a photo of fingers and palm'],['a photo of an elbow','a photo of a forearm','a photo of a mobile phone']),
 'cup':(['a photo of a drinking cup','a photo of a mug','a photo of a glass'],['a photo of a bottle','a photo of a plate']),
 'tray':(['a photo of a serving tray','a photo of a wooden tray'],['a photo of a plate','a photo of a table'])}
CB=['crop_x1','crop_y1','crop_x2','crop_y2']
for p in ['inputs','features','results/oof','results/submissions','models','logs','audit']:(A/p).mkdir(parents=True,exist_ok=True)
def main():
    global A
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['oof','test','pool','manual'],required=True);p.add_argument('--limit',type=int);p.add_argument('--output-root');args=p.parse_args()
    if args.output_root:
        A=ROOT/args.output_root
        for sub in ['inputs','features','logs']:(A/sub).mkdir(parents=True,exist_ok=True)
    assert torch.backends.mps.is_available();torch.set_num_threads(3);cv2.setNumThreads(1)
    q,te,h=inputs();rows=q if args.split=='oof' else te;rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv').set_index('id')
    if args.split=='pool':rows=pd.read_csv(A/'inputs/pool_records.csv');rec=rows.set_index('id')
    if args.split=='manual':rows=pd.read_csv(A/'inputs/manual_records.csv');rec=rows.set_index('id')
    path=A/f'features/{args.split}.jsonl';done={json.loads(l)['id'] for l in path.read_text().splitlines()} if path.exists() else set();rows=rows[~rows.id.isin(done)]
    if args.limit:rows=rows.head(args.limit)
    modelpath=ROOT/'rascar_v50/model_cache/owlv2';processor=Owlv2Processor.from_pretrained(str(modelpath),local_files_only=True);model=Owlv2ForObjectDetection.from_pretrained(str(modelpath),local_files_only=True).eval().to('mps')
    @lru_cache(maxsize=12)
    def image(fid):return cv2.resize(cv2.imread(str(D/f'{fid}.jpg')),(1280,720))
    start=time.time()
    for n,(_,r) in enumerate(rows.iterrows()):
        cb=rec.loc[r.id,CB].to_numpy(int);crop=image(r.frame_id)[cb[1]:cb[3],cb[0]:cb[2],::-1];pos,neg=PROMPTS[r.class_name];data=processor(text=[pos+neg],images=Image.fromarray(crop),return_tensors='pt');data={k:v.to('mps') for k,v in data.items()}
        with torch.inference_mode():out=model(**data)
        logits=out.logits[0].cpu().numpy();prob=1/(1+np.exp(-np.clip(logits,-80,80)));raw=out.pred_boxes[0].cpu().numpy();box=np.c_[raw[:,:2]-raw[:,2:]/2,raw[:,:2]+raw[:,2:]/2]*max(crop.shape[:2])+np.tile(cb[:2],2);obj=out.objectness_logits[0].sigmoid().cpu().numpy();positive=prob[:,:len(pos)].max(1);negative=prob[:,len(pos):].max(1);overlap=iou(r[B],box)[0]
        # Verify padding-to-square conversion against the installed official processor.
        if n==0:
            check=processor.post_process_object_detection(out,threshold=.1,target_sizes=torch.tensor([[crop.shape[0],crop.shape[1]]],device='mps'))[0];m=prob.max(1)>.1
            assert np.max(abs(check['boxes'].cpu().numpy()+np.tile(cb[:2],2)-box[m]),initial=0)<1e-3
        center=(box[:,:2]+box[:,2:])/2;valid_center=((center>=cb[:2])&(center<=cb[2:])).all(1)
        box=np.clip(box,np.tile(cb[:2],2),np.tile(cb[2:],2));positive=np.where(valid_center,positive,0.);negative=np.where(valid_center,negative,0.);overlap=iou(r[B],box)[0]
        local=overlap>=.3;best=int(np.argmax(positive*overlap));features=dict(pos_iou_support=float((positive*overlap).max()),neg_iou_support=float((negative*overlap).max()),positive_local=float(positive[local].max()) if local.any() else 0.,negative_local=float(negative[local].max()) if local.any() else 0.,semantic_margin=float(positive[best]-negative[best]),objectness_local=float(obj[local].max()) if local.any() else 0.,positive_max=float(positive.max()))
        refs=[];keep=[]
        for j in np.argsort(positive)[::-1]:
            if positive[j]<.03:break
            if keep and iou(box[j],box[keep]).max()>.5:continue
            keep.append(j);refs.append(dict(xyxy=box[j].tolist(),positive=float(positive[j]),negative=float(negative[j]),objectness=float(obj[j]),overlap=float(overlap[j])))
            if len(keep)>=30:break
        for cut in [.05,.1,.2]:
            candidates=[v for v in refs if v['positive']>=cut];ref=max(candidates,key=lambda x:x['overlap']) if candidates else None;tag=f'c{str(cut).replace(".","")}'
            features[f'iou_{tag}']=ref['overlap'] if ref else 0.;features[f'confidence_{tag}']=ref['positive'] if ref else 0.;features[f'missing_{tag}']=float(ref is None)
        selected=[v for v in refs if v['positive']>=.1];bestref=max(selected,key=lambda x:x['overlap']) if selected else None
        if bestref:
            delta=(np.array(bestref['xyxy'])-r[B].to_numpy(float))/np.tile(np.maximum(r[['x2','y2']].to_numpy(float)-r[['x1','y1']].to_numpy(float),1),2)
        else:delta=np.zeros(4)
        features.update(dict(zip(['delta_'+v for v in B],delta)));features['detector_confidence']=float(r.confidence)
        result=dict(id=r.id,frame_id=r.frame_id,day=r.day,split=args.split,class_name=r.class_name,crop=cb.tolist(),features=features,references=refs)
        with path.open('a') as f:f.write(json.dumps(result)+'\n')
        if n%20==0:print('OWL',args.split,n,len(rows),round(time.time()-start,1),'refs',len(refs),flush=True)
        del out,data
    print('DONE',args.split,len(rows),round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
