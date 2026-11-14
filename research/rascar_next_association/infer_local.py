"""Additional hand inference on expanded, model-detected person crops.
All crop selection is frozen and target-free, equally applied on train/test.
"""
from infer_bpj import *
import copy
from torchvision.ops import nms

def main():
    torch.set_num_threads(3)
    heads={x['frame_id']:x for x in map(json.loads,(A/'features/head.jsonl').read_text().splitlines())}
    hands={x['frame_id']:x for x in map(json.loads,(A/'features/hand.jsonl').read_text().splitlines())}
    target=A/'features/hand_local.jsonl';done=set()
    if target.exists():done={json.loads(s)['frame_id'] for s in target.read_text().splitlines()}
    model=attempt_load(str(A/'models/bh_hand_s_1536_e100_best.pt'),map_location='cpu');decode=author_decoder();start=time.time()
    protocol=dict(source='head-model bodies confidence >=.3 with linked head confidence >=.3',expansion=1.25,crop_nms=.7,min_native_side=40,max_crops=12,imgsz=1024,hand_merge_nms=.5,trained=False,uses_test_labels=False)
    (A/'inputs/local_protocol.json').write_text(json.dumps(protocol,indent=2))
    for frame in sorted(heads):
        if frame in done:continue
        t=time.time();hr=heads[frame];rec=copy.deepcopy(hands[frame]);im=cv2.imread(str(ROOT/'rascar_data_full'/f'{frame}.jpg'));assert im is not None
        H,W=im.shape[:2];scale=np.array([W/1280,H/720,W/1280,H/720]);boxes=[];confs=[]
        for body in hr['bodies']:
            confirmed=any(p['confidence']>=.3 and any(v['body_id']==body['instance_id'] for v in p['associations']) for p in hr['parts'])
            if body['confidence']<.3 or not confirmed:continue
            b=np.array(body['xyxy'])*scale;center=(b[:2]+b[2:])/2;wh=(b[2:]-b[:2])*1.25;b=np.r_[np.maximum(0,np.floor(center-wh/2)),np.minimum([W,H],np.ceil(center+wh/2))].astype(int)
            if (b[2:]-b[:2]).min()<40:continue
            boxes.append(b);confs.append(body['confidence'])
        keep=nms(torch.tensor(np.array(boxes),dtype=torch.float32),torch.tensor(confs),.7).tolist()[:12] if boxes else []
        cropmeta=[]
        for j in keep:
            x1,y1,x2,y2=boxes[j];crop=im[y1:y2,x1:x2];local=infer(model,crop,'hand',decode,1024,crop.shape[:2]);offset=np.array([x1,y1,x1,y1]);nb=len(rec['bodies']);np0=len(rec['parts'])
            for b in local['bodies']:
                b['instance_id']+=nb;b['xyxy']=((np.array(b['xyxy'])+offset)/scale).tolist();b['predicted_part_centers']=((np.array(b['predicted_part_centers'])+[x1,y1])/scale[:2]).tolist();b['crop_id']=len(cropmeta);rec['bodies'].append(b)
            for p in local['parts']:
                p['part_id']+=np0;p['xyxy']=((np.array(p['xyxy'])+offset)/scale).tolist();p['crop_id']=len(cropmeta)
                for link in p['associations']:link['body_id']+=nb;link['part_id']+=np0;link['center_distance']/=scale[0]
                rec['parts'].append(p)
            cropmeta.append(dict(native_xyxy=[int(v) for v in boxes[j]],bodies=len(local['bodies']),parts=len(local['parts'])))
        if rec['parts']:
            b=torch.tensor([p['xyxy'] for p in rec['parts']],dtype=torch.float32);c=torch.tensor([p['confidence'] for p in rec['parts']]);keep=nms(b,c,.5).tolist();rec['parts']=[rec['parts'][j] for j in keep]
            for j,p in enumerate(rec['parts']):
                p['part_id']=j
                for link in p['associations']:link['part_id']=j
        rec.update(crops=cropmeta,local_seconds=time.time()-t,unassigned_part_ids=[p['part_id'] for p in rec['parts'] if not p['associations']])
        with target.open('a') as f:f.write(json.dumps(rec)+'\n')
        print(frame,'crops',len(cropmeta),'hands',len(hands[frame]['parts']),'->',len(rec['parts']),'sec',round(time.time()-t,2),flush=True)
    print('FINISHED',round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
