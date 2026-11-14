from common import *
import sys,torch
from PIL import Image
from transformers import Sam3Model,Sam3Processor
from kaggle_secrets import UserSecretsClient
from huggingface_hub import model_info

kind=sys.argv[1];device='cuda:'+sys.argv[2];torch.set_num_threads(2)
token=UserSecretsClient().get_secret('HF_TOKEN');revision=model_info('facebook/sam3',token=token).sha
model=Sam3Model.from_pretrained('facebook/sam3',revision=revision,token=token,dtype=torch.float16).to(device).eval()
processor=Sam3Processor.from_pretrained('facebook/sam3',revision=revision,token=token);del token
paths=pd.read_csv(A/'image_paths.csv').set_index('frame_id').path.to_dict()
rows=pd.concat([q.assign(split='train'),te.assign(split='test')],ignore_index=True)
target='person' if kind=='person' else 'nude_hands';rows=rows[rows.class_name.eq(target)]
prompts=['person','head'] if kind=='person' else ['hand','forearm']
out=[];summary=[]
for n,r in enumerate(rows.itertuples()):
    im=Image.open(paths[r.frame_id]).convert('RGB');b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];center=(b[:2]+b[2:])/2
    scale,minimum=(1.8,256) if kind=='person' else (4,160)
    cropwh=np.maximum(wh*scale,minimum);roi=np.r_[center-cropwh/2,center+cropwh/2];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720);roi=roi.astype(int)
    crop=im.crop(tuple(roi));inputs=processor(images=crop,return_tensors='pt').to(device,torch.float16)
    with torch.inference_mode():vision=model.get_vision_features(pixel_values=inputs.pixel_values)
    for prompt in prompts:
        textinputs=processor(text=prompt,return_tensors='pt').to(device)
        with torch.inference_mode():outputs=model(vision_embeds=vision,**textinputs)
        res=processor.post_process_instance_segmentation(outputs,threshold=.15,mask_threshold=.5,target_sizes=inputs.get('original_sizes').tolist())[0]
        summary.append(dict(id=r.id,frame_id=r.frame_id,split=r.split,prompt=prompt,objects=len(res['boxes'])))
        for j,(bb,score,mask) in enumerate(zip(res['boxes'].cpu().numpy(),res['scores'].cpu().numpy(),res['masks'].cpu().numpy())):
            yy,xx=np.where(mask.astype(bool));mb=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1]) if len(xx) else bb
            qb=(b-np.tile(roi[:2],2)).astype(int);qb[[0,2]]=np.clip(qb[[0,2]],0,crop.width);qb[[1,3]]=np.clip(qb[[1,3]],0,crop.height)
            within=float(mask[qb[1]:qb[3],qb[0]:qb[2]].sum());total=float(mask.sum())
            out.append(dict(query_id=r.id,frame_id=r.frame_id,split=r.split,prompt=prompt,confidence=float(score),reference_id=f'sam16_{r.id}_{prompt}_{j}',
                mask_in_query=within/max(total,1),mask_query_coverage=within/max(np.prod(qb[2:]-qb[:2]),1),mask_area=total,
                **dict(zip(B,bb+np.tile(roi[:2],2))),**dict(zip(['mask_'+c for c in B],mb+np.tile(roi[:2],2)))))
    if n%10==0:print('SAM16_PROGRESS',kind,n,len(rows),flush=True)
pd.DataFrame(out).to_csv(A/f'local_sam_{kind}.csv',index=False)
pd.DataFrame(summary).to_csv(A/f'local_sam_{kind}_coverage.csv',index=False)
(A/f'local_sam_{kind}_manifest.json').write_text(json.dumps(dict(model='facebook/sam3',revision=revision,prompts=prompts,scale=scale,minimum=minimum,rows=len(rows),frozen=True),indent=2))
print('SAM16_DONE',kind,len(out),flush=True)
