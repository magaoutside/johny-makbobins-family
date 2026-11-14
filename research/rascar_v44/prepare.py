from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v43'))
from common43 import *
A=ROOT/'rascar_v44'
import torch,cv2
from transformers import AutoModel
from PIL import Image,ImageDraw,ImageFont
CB=['crop_x1','crop_y1','crop_x2','crop_y2']
def occupancy(boxes,cb):
    step=(cb[2:]-cb[:2])/32;xx=cb[0]+np.arange(32)*step[0];yy=cb[1]+np.arange(32)*step[1];mask=np.zeros((32,32),np.float32)
    for b in boxes:
        wx=np.maximum(0,np.minimum(xx+step[0],b[2])-np.maximum(xx,b[0]))/step[0];wy=np.maximum(0,np.minimum(yy+step[1],b[3])-np.maximum(yy,b[1]))/step[1];mask=np.maximum(mask,wy[:,None]*wx[None,:])
    return mask
def main():
    review=pd.read_csv(A/'pool_review/reviews.csv');boxes=pd.read_csv(A/'pool_review/boxes.csv');new=[];rng=np.random.default_rng(4401)
    for _,v in review[review.status.eq('accepted_silver')].iterrows():
        cb=v[CB].to_numpy(float);cc=(cb[:2]+cb[2:])/2;wh=cb[2:]-cb[:2]
        for aug in range(3 if v.role=='train' else 1):
            crop=cb if aug==0 else np.r_[cc+np.array([-1,1])*(aug-1)*4-wh*.375,cc+np.array([-1,1])*(aug-1)*4+wh*.375]
            new.append(dict(id=f'pool44_{int(v.review_index)}_{aug}',frame_id=v.frame_id,day=pd.to_datetime(str(v.day)).strftime('%Y-%m-%d'),class_name='nude_hands',split=v.role,review_index=int(v.review_index),uncertainty=v.edge_uncertainty_px,**dict(zip(B,v[B])),**dict(zip(CB,crop))))
    rec=pd.DataFrame(new);rec.to_csv(A/'inputs/pool_records.csv',index=False);targets=[];valid=[]
    for _,v in rec.iterrows():
        bb=boxes[boxes.review_index.eq(v.review_index)][B].to_numpy(float);cb=v[CB].to_numpy(float);u=float(v.uncertainty)
        target=occupancy(bb,cb)
        if len(bb):
            inner=bb+np.array([u,u,-u,-u]);outer=bb+np.array([-u,-u,u,u]);amb=(occupancy(outer,cb)-occupancy(inner,cb))>.01;mask=(~amb).astype(np.float32)
        else:mask=occupancy([v[B].to_numpy(float)+np.array([-3,-3,3,3])],cb)
        targets.append(target);valid.append(mask)
    np.save(A/'features/pool_targets.npy',targets);np.save(A/'features/pool_valid.npy',valid)
    torch.set_num_threads(4);device='mps';assert torch.backends.mps.is_available();m=AutoModel.from_pretrained(str(ROOT/'rascar_v31/model_cache/dinov2-small'),local_files_only=True).eval().to(device)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    z=np.lib.format.open_memmap(A/'features/pool_grids.npy',mode='w+',dtype=np.float16,shape=(len(rec),384,32,32));start=time.time()
    with torch.inference_mode():
      for off in range(0,len(rec),8):
        arr=[]
        for _,v in rec.iloc[off:off+8].iterrows():
            im=cv2.resize(cv2.imread(str(D/f'{v.frame_id}.jpg')),(1280,720));x1,y1,x2,y2=v[CB].to_numpy(int);arr.append(cv2.resize(im[y1:y2,x1:x2,::-1],(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
        x=torch.tensor(np.stack(arr),dtype=torch.float32,device=device)/255;f=m(pixel_values=(x-mean)/std).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,32,32);z[off:off+len(arr)]=f.cpu().numpy().astype(np.float16)
    z.flush();(A/'features/manifest.json').write_text(json.dumps(dict(rows=len(rec),seconds=time.time()-start,source='V31 pinned facebook/dinov2-small',record_sha256=hashlib.sha256((A/'inputs/pool_records.csv').read_bytes()).hexdigest(),test_annotations=False),indent=2))
    # Review the new boxes over original pool crops before fitting.
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',17)
    for off in range(0,len(review),6):
      sheet=Image.new('RGB',(1000,1350),'#171b23');dr=ImageDraw.Draw(sheet)
      for k,(_,v) in enumerate(review.iloc[off:off+6].iterrows()):
        x=k%2*500;y=k//2*450;cb=v[CB].to_numpy(float);im=Image.open(D/f'{v.frame_id}.jpg').convert('RGB').resize((1280,720)).crop(tuple(cb.astype(int))).resize((384,384),Image.Resampling.NEAREST);sheet.paste(im,(x+55,y+55));scale=np.tile(384/(cb[2:]-cb[:2]),2);offset=np.tile([x+55,y+55],2)
        for _,g in boxes[boxes.review_index.eq(v.review_index)].iterrows():dr.rectangle(tuple((g[B].to_numpy(float)-np.tile(cb[:2],2))*scale+offset),outline='#56ff85',width=2)
        dr.rectangle(tuple((v[B].to_numpy(float)-np.tile(cb[:2],2))*scale+offset),outline='#ffde62',width=1)
        dr.text((x+10,y+5),f'#{int(v.review_index)} {v.role} {v.status}',font=font,fill='white')
        dr.text((x+10,y+28),'green: unofficial manual bounds / yellow: original',font=font,fill='#aaaaaa')
      sheet.save(A/f'pool_review/annotated_{off//6:02d}.jpg',quality=96)
    print('DONE',len(rec),time.time()-start,flush=True)
if __name__=='__main__':main()
