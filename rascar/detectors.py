"""Frozen independent WeVis and multi-view pose evidence, regenerated from RGB."""
from pathlib import Path
import json, re, time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from .core import BBOX as B, queries


def paths(raw):
    result={p.stem:p for p in Path(raw).rglob('*.jpg')}
    if not result: raise FileNotFoundError(f'No JPG images below {raw}')
    return result


def wevis(data, raw, weights, out, device, mode):
    from transformers import AutoProcessor,AutoModelForImageTextToText
    torch.set_num_threads(2)
    rows=queries(data);files=paths(raw);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cache=out/f'wevis_{mode}_cache';cache.mkdir(exist_ok=True)
    pixel_limit=802816 if mode in ['heads','hands'] else 921600
    processor=AutoProcessor.from_pretrained(str(Path(weights)/'wevis'),min_pixels=16384,max_pixels=pixel_limit,trust_remote_code=True,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(str(Path(weights)/'wevis'),dtype=torch.float16,attn_implementation='sdpa',trust_remote_code=True,local_files_only=True).to(device).eval()
    items=[]
    if mode=='heads':
        items=[dict(id=fid,frame_id=fid,roi=[0,0,1280,720]) for fid in rows.frame_id.unique()]
    elif mode=='full':
        manual=pd.read_csv(Path(data)/'manual_boxes.csv')
        ids=list(dict.fromkeys(manual.frame_id.tolist()+rows.loc[rows.split.eq('test')&rows.class_name.eq('person'),'frame_id'].tolist()))
        items=[dict(id=fid,frame_id=fid,roi=[0,0,1280,720]) for fid in ids]
    else:
        select=rows.class_name.eq('nude_hands') if mode=='hands' else rows.class_name.ne('person')
        for _,r in rows[select].iterrows():
            b=r[B].to_numpy(float);center=(b[:2]+b[2:])/2
            if mode=='hands':
                wh=np.maximum((b[2:]-b[:2])*4,160)
                roi=np.r_[center-wh/2,center+wh/2];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720)
                roi=roi.astype(int)
            else:
                wh=np.minimum(np.maximum([320,256],4*(b[2:]-b[:2])),[1280,720])
                origin=np.clip(center-wh/2,[0,0],np.array([1280,720])-wh)
                roi=np.r_[np.floor(origin),np.ceil(origin+wh)].astype(int)
            items.append(dict(id=r.id,frame_id=r.frame_id,class_name=r.class_name,roi=roi.tolist()))
    tokens={'full':4096,'zoom':1024,'heads':384,'hands':384}[mode]
    boxes=[]
    for n,item in enumerate(items):
        dest=cache/(item['id']+'.json')
        if not dest.exists():
            im=Image.open(files[item['frame_id']]).convert('RGB')
            if im.size!=(1280,720):raise ValueError(f'Expected original 1280x720 image: {item["frame_id"]}')
            im=im.crop(tuple(item['roi']));original=im.size
            if mode!='full':
                factor=min(1120/im.width,896/im.height)
                im=im.resize((round(im.width*factor),round(im.height*factor)),Image.Resampling.BICUBIC)
            desc={'full':'person. cup. bare human hand. tray. table','heads':'human head',
                  'hands':'bare human hand, including the palm and fingers but excluding the forearm'}.get(mode,{'nude_hands':'bare human hand'}.get(item.get('class_name'),item.get('class_name')))
            prompt='Locate all the instances that match the following description: '+desc+'.'
            messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':prompt}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            feed=processor(text=[text],images=[im],padding=True,return_tensors='pt').to(device);start=time.time()
            with torch.inference_mode():pred=model.generate(**feed,max_new_tokens=tokens,repetition_penalty=1.05,use_cache=True,do_sample=False)
            response=processor.batch_decode(pred[:,feed['input_ids'].shape[1]:],skip_special_tokens=False,clean_up_tokenization_spaces=False)[0]
            record=dict(**item,response=response,prompt=prompt,seconds=time.time()-start,new_tokens=int(pred.shape[1]-feed['input_ids'].shape[1]),crop_size=original)
            dest.write_text(json.dumps(record))
        r=json.loads(dest.read_text())
        if r['new_tokens']>=tokens:raise ValueError(f'Truncated model response: {dest}')
        parts=re.findall(r'<ref>(.*?)</ref>(.*?)(?=<ref>|$)',r['response'],re.S) if mode=='full' else [(item.get('class_name',mode),r['response'])]
        count=0
        for cls,body in parts:
            cls={'bare human hand':'nude_hands','bare human hands':'nude_hands'}.get(cls.strip().lower(),cls.strip().lower())
            for part in re.findall(r'<box>(.*?)</box>',body,re.S):
                vals=[int(x) for x in re.findall(r'<(\d{1,4})>',part)]
                if len(vals)%4:raise ValueError(f'Malformed detection: {dest}')
                for i in range(0,len(vals),4):
                    b=np.array(vals[i:i+4],float)
                    if (b<0).any() or (b>1000).any() or min(b[2:]-b[:2])<=0:continue
                    b=b/1000*np.tile(r['crop_size'],2)+np.tile(item['roi'][:2],2)
                    boxes.append(dict(id=item['id'],query_id=item['id'],frame_id=item['frame_id'],class_name=cls,reference_id=f'{mode}_{item["id"]}_{count}',confidence=1.,**dict(zip(B,b))))
                    count+=1
        print(f'WeVis {mode}: {n+1}/{len(items)}',flush=True)
    name={'heads':'heads','hands':'hands','full':'wevis_full','zoom':'zoom_boxes'}[mode]
    result=pd.DataFrame(boxes,columns=['id','query_id','frame_id','class_name','reference_id','confidence']+B)
    result.to_csv(out/f'{name}.csv',index=False)


def pose(data, raw, weights, out, device):
    from ultralytics import YOLO
    torch.set_num_threads(2)
    files=paths(raw);rows=queries(data);manual=pd.read_csv(Path(data)/'manual_boxes.csv')
    frames=list(dict.fromkeys(manual.frame_id.tolist()+rows.frame_id.tolist()))
    model=YOLO(str(Path(weights)/'pose/yolo26m-pose.pt'));result=[]
    def inverse(xy,turn,w,h):
        x,y=xy[...,0].copy(),xy[...,1].copy()
        if turn==1:return np.stack([w-y,x],axis=-1)
        if turn==3:return np.stack([y,h-x],axis=-1)
        return xy
    for i,fid in enumerate(frames):
        im=Image.open(files[fid]).convert('RGB')
        if im.size!=(1280,720):raise ValueError(f'Expected original 1280x720 image: {fid}')
        specs=[(0,0,1280,720,t) for t in [0,1,3]]+[(x,y,x+800,y+500,0) for x in [0,480] for y in [0,220]]
        for vi,(x1,y1,x2,y2,turn) in enumerate(specs):
            crop=im.crop((x1,y1,x2,y2));w,h=crop.size
            r=model.predict(Image.fromarray(np.rot90(np.asarray(crop),turn).copy()),imgsz=960,conf=.08,iou=.5,device=device,verbose=False)[0]
            for j,(box,conf,kp) in enumerate(zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.keypoints.data.cpu().numpy())):
                corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]])
                xy=inverse(corners,turn,w,h)+[x1,y1];bb=np.r_[xy.min(0),xy.max(0)];kxy=inverse(kp[:,:2],turn,w,h)+[x1,y1]
                row=dict(frame_id=fid,view=vi,turn=turn,confidence=float(conf),reference_id=f'pose16_{fid}_{vi}_{j}',**dict(zip(B,bb)))
                for k in range(17):row.update({f'kp{k}_x':float(kxy[k,0]),f'kp{k}_y':float(kxy[k,1]),f'kp{k}_conf':float(kp[k,2])})
                result.append(row)
        if i%10==0:print(f'Pose: {i}/{len(frames)}',flush=True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(result).to_csv(out/'poses.csv',index=False)
