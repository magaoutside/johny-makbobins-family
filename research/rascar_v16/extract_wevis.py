from common import *
import sys,re,time,torch,os
from PIL import Image
from transformers import AutoProcessor,AutoModelForImageTextToText
os.environ['TOKENIZERS_PARALLELISM']='false';torch.set_num_threads(2)
task=sys.argv[1];device='cuda:'+sys.argv[2]
source=A/'source_v8/sam31_v8/zoom_test_job.json';revision=json.loads(source.read_text())['revision']
name='tencent/WeVisOne-VL-4B'
processor=AutoProcessor.from_pretrained(name,revision=revision,min_pixels=16384,max_pixels=802816,trust_remote_code=True)
model=AutoModelForImageTextToText.from_pretrained(name,revision=revision,dtype=torch.float16,attn_implementation='sdpa',trust_remote_code=True).to(device).eval()
for p in model.parameters():p.requires_grad_(False)
paths=pd.read_csv(A/'image_paths.csv').set_index('frame_id').path.to_dict()
rows=pd.concat([q.assign(split='train'),te.assign(split='test')],ignore_index=True)
if task=='heads':
    items=[dict(id=fid,frame_id=fid,roi=[0,0,1280,720]) for fid in rows.frame_id.unique()]
    description='human head'
else:
    items=[]
    for r in rows[rows.class_name.eq('nude_hands')].itertuples():
        b=np.array([getattr(r,c) for c in B]);wh=np.maximum((b[2:]-b[:2])*4,160);center=(b[:2]+b[2:])/2
        roi=np.r_[center-wh/2,center+wh/2];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720)
        items.append(dict(id=r.id,frame_id=r.frame_id,roi=roi.astype(int).tolist()))
    description='bare human hand, including the palm and fingers but excluding the forearm'
cache=A/('wevis_'+task+'_cache');cache.mkdir(exist_ok=True)
for n,item in enumerate(items):
    dest=cache/(item['id']+'.json')
    if dest.exists():continue
    im=Image.open(paths[item['frame_id']]).convert('RGB').crop(tuple(item['roi']));original=im.size
    factor=min(1120/im.width,896/im.height);im=im.resize((round(im.width*factor),round(im.height*factor)),Image.Resampling.BICUBIC)
    prompt='Locate all the instances that match the following description: '+description+'.'
    messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':prompt}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],images=[im],padding=True,return_tensors='pt').to(device);start=time.time()
    with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=384,repetition_penalty=1.05,use_cache=True,do_sample=False)
    response=processor.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=False,clean_up_tokenization_spaces=False)[0]
    dest.write_text(json.dumps(dict(**item,response=response,prompt=prompt,seconds=time.time()-start,new_tokens=int(out.shape[1]-inputs['input_ids'].shape[1]),crop_size=original,revision=revision)))
    print('WEVIS16',task,n+1,len(items),round(time.time()-start,1),response[:100],flush=True)
    del inputs,out
(A/f'wevis_{task}_manifest.json').write_text(json.dumps(dict(model=name,revision=revision,prompt=prompt,items=len(items),frozen=True,max_new_tokens=384),indent=2))
print('WEVIS16_DONE',task,len(items),flush=True)
