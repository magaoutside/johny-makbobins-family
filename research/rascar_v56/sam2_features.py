"""Independent prompted SAM2 masks and response to bbox perturbations.

Frozen image model. No manual boxes, error labels, or test reviews enter features.
"""
from pathlib import Path
import sys,os,json,time,hashlib,argparse
A=Path(__file__).resolve().parent;ROOT=A.parent
os.environ.setdefault('HF_HOME',str(A/'cache/hf'));os.environ.setdefault('MPLCONFIGDIR',str(A/'cache/mpl'))
sys.path.insert(0,str(A/'vendor_sam'))
import numpy as np,pandas as pd,torch,cv2
from transformers import Sam2Model,Sam2Processor
from PIL import Image

B=['x1','y1','x2','y2'];CB=['crop_x1','crop_y1','crop_x2','crop_y2']
def iou(a,b):
    a=np.array(a,float);b=np.array(b,float);z=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return z/max((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-z,1e-9)

def prompts(b):
    wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2
    return [b,np.r_[cc-.425*wh,cc+.425*wh],np.r_[cc-.575*wh,cc+.575*wh],b+np.tile([-.12*wh[0],0],2),b+np.tile([.12*wh[0],0],2)]

def summarize(mask,logits,confidence,q,crop):
    yy,xx=np.where(mask);out=dict(confidence=float(confidence),empty=float(len(xx)==0))
    q=q-np.tile(crop[:2],2);wh=np.maximum(q[2:]-q[:2],1);area=wh.prod();ch,cw=mask.shape
    qx1,qy1,qx2,qy2=np.clip(np.rint(q).astype(int),[0,0,0,0],[cw,ch,cw,ch]);inside=mask[qy1:qy2,qx1:qx2].sum()
    if len(xx):bb=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],float)
    else:bb=np.zeros(4)
    n,lab,stats,cent=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    inter=np.maximum(0,np.minimum(q[2:],bb[2:])-np.maximum(q[:2],bb[:2])).prod();barea=max((bb[2:]-bb[:2]).prod(),1)
    out.update(bbox_iou=iou(q,bb),bbox_cover_query=inter/area,bbox_cover_ref=inter/barea,
        mask_query_coverage=float(inside/area),mask_inside_fraction=float(inside/max(mask.sum(),1)),mask_fill=float(mask.sum()/barea),mask_relative_area=float(mask.sum()/area),
        components=float((stats[1:,cv2.CC_STAT_AREA]>=max(4,area*.01)).sum()),edge_contact=float(mask[0].mean()+mask[-1].mean()+mask[:,0].mean()+mask[:,-1].mean()),
        uncertainty=float((abs(logits)<1).mean()),logit_mean=float(logits.mean()),
        left=float((q[0]-bb[0])/wh[0]),top=float((q[1]-bb[1])/wh[1]),right=float((q[2]-bb[2])/wh[0]),bottom=float((q[3]-bb[3])/wh[1]))
    return out,bb+np.tile(crop[:2],2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--modes',nargs='+',default=['full','crop']);p.add_argument('--limit',type=int,default=0);p.add_argument('--device',choices=['cpu','mps'],default='cpu');args=p.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    model,loading=Sam2Model.from_pretrained(str(A/'models/sam2.1-hiera-tiny'),local_files_only=True,output_loading_info=True)
    assert not loading['missing_keys'],loading['missing_keys']
    (A/'audit/sam2_weight_loading.json').write_text(json.dumps(loading,indent=2))
    model=model.eval().to(args.device)
    processor=Sam2Processor.from_pretrained(str(A/'models/sam2.1-hiera-tiny'),local_files_only=True)
    rows=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');rows=rows[rows.split.isin(['oof','test'])].reset_index(drop=True)
    start=time.time();allrows=[]
    for mode in args.modes:
        output=A/f'features/sam2_{mode}.csv';done=set()
        if output.exists():allrows=pd.read_csv(output).to_dict('records');done={r['id'] for r in allrows}
        else:allrows=[]
        groups=list(rows.groupby('frame_id',sort=False)) if mode=='full' else [(r.frame_id,pd.DataFrame([r])) for _,r in rows.iterrows()]
        for n,(frame,g) in enumerate(groups):
            if args.limit and n>=args.limit:break
            g=g[~g.id.isin(done)]
            if not len(g):continue
            im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{frame}.jpg')),(1280,720))[:,:,::-1].copy()
            crop=np.array([0,0,1280,720]) if mode=='full' else g.iloc[0][CB].to_numpy(int)
            im=im[crop[1]:crop[3],crop[0]:crop[2]];boxes=[]
            for _,r in g.iterrows():boxes.extend(prompts(r[B].to_numpy(float)))
            boxes=np.array(boxes)-np.tile(crop[:2],2);boxes=np.clip(boxes,[0,0,0,0],[im.shape[1],im.shape[0],im.shape[1],im.shape[0]])
            feed=processor(images=Image.fromarray(im),input_boxes=[boxes.tolist()],return_tensors='pt')
            with torch.inference_mode():
                embeddings=model.get_image_embeddings(feed['pixel_values'].to(args.device))
                masks=[];scores=[]
                # Decoder chunks bound RAM on crowded frames.
                for off in range(0,len(boxes),5):
                    pred=model(image_embeddings=embeddings,input_boxes=feed['input_boxes'][:,off:off+5].to(args.device),multimask_output=True)
                    masks.extend(pred.pred_masks[0].cpu().numpy());scores.extend(pred.iou_scores[0].cpu().numpy())
            for j,(_,r) in enumerate(g.iterrows()):
                features=[];bbs=[];binary=[];q=r[B].to_numpy(float)
                for k in range(5):
                    confidence=scores[j*5+k];best=int(np.argmax(confidence));low=masks[j*5+k][best]
                    logits=cv2.resize(low,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_LINEAR);mask=logits>0
                    f,bb=summarize(mask,logits,confidence[best],q,crop);features.append(f);bbs.append(bb)
                    binary.append(cv2.resize(mask.astype(np.uint8),(128,128),interpolation=cv2.INTER_NEAREST).astype(bool))
                out=dict(id=r.id)
                for key in features[0]:
                    values=np.array([f[key] for f in features]);out[f'sam2_{mode}_{key}']=features[0][key]
                    if key in ['bbox_iou','confidence','mask_query_coverage','mask_inside_fraction','mask_relative_area','left','top','right','bottom']:
                        out[f'sam2_{mode}_{key}_mean']=float(values.mean());out[f'sam2_{mode}_{key}_std']=float(values.std())
                out[f'sam2_{mode}_bbox_stability']=float(np.mean([iou(bbs[0],x) for x in bbs[1:]]))
                out[f'sam2_{mode}_mask_stability']=float(np.mean([(binary[0]&x).sum()/max((binary[0]|x).sum(),1) for x in binary[1:]]))
                allrows.append(out)
            if n%10==0:pd.DataFrame(allrows).to_csv(output,index=False);print('SAM2',mode,n,len(groups),'rows',len(allrows),'sec',round(time.time()-start,1),flush=True)
        pd.DataFrame(allrows).to_csv(output,index=False)
        assert args.limit or len(allrows)==len(rows)
    (A/'features/sam2_extraction.json').write_text(json.dumps(dict(model='facebook/sam2.1-hiera-tiny',mode=args.modes,seconds=time.time()-start,labels_used=False,query_perturbations=['original','scale0.85','scale1.15','left0.12','right0.12'],selection='Highest predicted mask-IoU among 3 outputs per box, without GT'),indent=2))

if __name__=='__main__':main()
