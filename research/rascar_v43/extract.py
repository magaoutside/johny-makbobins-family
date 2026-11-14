from common43 import *
import torch,cv2,time
from torchvision.ops import roi_align
from torchvision.models.detection.transform import resize_boxes

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);assert torch.backends.mps.is_available()
    device='mps';m=detector(device);r=pd.read_csv(A/'inputs/records.csv');n=len(r)
    # MultiScaleRoIAlign native CPU kernel avoids unsupported MPS custom ops.
    maps=np.lib.format.open_memmap(A/'features/roi_maps.npy',mode='w+',dtype=np.float16,shape=(n,2,256,7,7))
    vectors=np.lib.format.open_memmap(A/'features/roi_vectors.npy',mode='w+',dtype=np.float32,shape=(n,2,1024))
    frozen=[];start=time.time()
    with torch.inference_mode():
      for fi,(fid,g) in enumerate(r.groupby('frame_id',sort=True)):
        im=cv2.resize(cv2.imread(str(D/f'{fid}.jpg')),(1280,720));im=torch.from_numpy(im[:,:,::-1].copy()).permute(2,0,1).float()/255
        images,_=m.transform([im.to(device)],None);feats=m.backbone(images.tensors)
        feats={k:v.cpu() for k,v in feats.items()}
        b=torch.tensor(g[B].to_numpy(),dtype=torch.float32);wh=b[:,2:]-b[:,:2];cc=(b[:,2:]+b[:,:2])/2
        span=torch.maximum(wh*1.6,torch.tensor([48.,48.]));context=torch.cat([cc-span/2,cc+span/2],1).clamp(min=0)
        context[:,[0,2]]=context[:,[0,2]].clamp(max=1280);context[:,[1,3]]=context[:,[1,3]].clamp(max=720)
        allboxes=torch.cat([b,context]);scaled=resize_boxes(allboxes,(720,1280),images.image_sizes[0])
        pooled=m.roi_heads.box_roi_pool(feats,[scaled],images.image_sizes)
        # Store complete spatial ROI maps to train the detector head later.
        for v in range(2):
          px=pooled[v*len(g):(v+1)*len(g)];maps[g.index,v]=px.numpy().astype(np.float16)
          outs=[]
          for j in range(0,len(px),32):outs.append(m.roi_heads.box_head(px[j:j+32].to(device)).cpu())
          embedding=torch.cat(outs);vectors[g.index,v]=embedding.numpy()
          if v==0:
            logits,deltas=m.roi_heads.box_predictor(embedding.to(device));probs=logits.softmax(1).cpu().numpy()
            decoded=m.roi_heads.box_coder.decode(deltas.cpu(),[scaled[:len(g)]]).reshape(len(g),91,4)
            decoded=resize_boxes(decoded.reshape(-1,4),images.image_sizes[0],(720,1280)).reshape(len(g),91,4).numpy()
            for j,(_,x) in enumerate(g.iterrows()):
              coco=1 if x.class_name=='person' else 47 if x.class_name=='cup' else 0
              bb=decoded[j,coco];ov=float(iou(x[B],bb)[0,0]);frozen.append(dict(id=x.id,class_name=x.class_name,coco_class=coco,coco_probability=float(probs[j,coco]),ref_iou=ov,**dict(zip(['ref_'+v for v in B],bb))))
        if fi%20==0:print('EXTRACT',fi,r.frame_id.nunique(),'elapsed',round(time.time()-start,1),flush=True)
        del images,feats,pooled
    maps.flush();vectors.flush();pd.DataFrame(frozen).set_index('id').loc[r.id].reset_index().to_csv(A/'features/frozen_coco.csv',index=False)
    (A/'features/manifest.json').write_text(json.dumps(dict(rows=n,frames=int(r.frame_id.nunique()),seconds=time.time()-start,torch=torch.__version__,device=device,roi_align='torchvision MultiScaleRoIAlign CPU float32',size=[800,1333],backbone_frozen=True,model_source=json.loads((A/'model_cache/source.json').read_text())),indent=2))
    print('COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
