"""Use only already reviewed pool boxes; never promote person presence to geometry."""
from prepare import *
from PIL import Image,ImageDraw

def main():
    p=pd.concat([pd.read_csv(ROOT/f'rascar_v22/inputs/approved_pseudo_boxes_{d}.csv') for d in DAYS],ignore_index=True)
    assert not set(p.frame_id)&set(h.frame_id)
    assert ((p.label_use.eq('geometry_and_presence')&p.visual_decision.eq('accept'))|(
        p.label_use.eq('presence_only_not_boundary_ground_truth')&p.visual_decision.eq('semantic_only'))).all()
    rng=np.random.default_rng(3304);rows=[]
    for j,r in p.iterrows():
        geometry=r.label_use=='geometry_and_presence';b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);center=(b[:2]+b[2:])/2
        assert r.class_name in (['cup','nude_hands'] if geometry else ['person'])
        # One pseudo crop per box, two per manual object in V31. Pseudo data
        # receives lower loss weight, and is sampled separately during fitting.
        cc=center+rng.uniform(-.2,.2,2)*wh;size=wh*2.6
        lo=np.maximum(0,np.floor(cc-size/2));hi=np.minimum([1280,720],np.ceil(cc+size/2))
        rows.append(dict(id=f'pseudo33_{r.id}',source_id=r.id,frame_id=r.frame_id,day=r.day,class_name=r.class_name,split='pool_geometry' if geometry else 'pool_presence',origin_id=r.id,label_use=r.label_use,**dict(zip(B,b)),**dict(zip(CB,np.r_[lo,hi]))))
    rec=pd.DataFrame(rows);rec['feature_index']=np.arange(len(rec));rec.to_csv(A/'inputs/pseudo_records.csv',index=False)
    torch.set_num_threads(3);cv2.setNumThreads(1)
    model=AutoModel.from_pretrained(str(V31/'model_cache/dinov2-small'),local_files_only=True,trust_remote_code=False).eval()
    dst=np.lib.format.open_memmap(A/'features/pseudo_grids.npy',mode='w+',dtype=np.float16,shape=(len(rec),384,32,32))
    mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1);start=time.time();sizes={}
    for off in range(0,len(rec),4):
        batch=[]
        for _,r in rec.iloc[off:off+4].iterrows():
            im=cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg'));assert im is not None;sizes[r.frame_id]=[im.shape[1],im.shape[0]]
            im=cv2.resize(im,(1280,720));b=r[CB].to_numpy(int);im=im[b[1]:b[3],b[0]:b[2],::-1]
            batch.append(cv2.resize(im,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
        x=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std
        with torch.inference_mode():z=model(pixel_values=x).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,32,32)
        dst[off:off+len(batch)]=z.numpy().astype(np.float16)
        if off%80==0:print('POOL',off,len(rec),round(time.time()-start,1),flush=True)
    dst.flush()
    (A/'inputs/pseudo_manifest.json').write_text(json.dumps(dict(boxes=len(rec),frames=int(rec.frame_id.nunique()),class_counts=rec.class_name.value_counts().to_dict(),source='V22 assistant_visual_review_nonofficial; not official GT',training_policy='Only fit day; geometry/quality loss at weight .3 for cups/hands; person presence loss only .3, zero boundary/quality loss.',original_image_sizes=sizes,coordinates='1280x720; pool images resized from native 640x360 before crop',test_annotations_used=False),indent=2))
    hands=rec[rec.class_name.eq('nude_hands')]
    for page,startrow in enumerate(range(0,len(hands),36)):
        canvas=Image.new('RGB',(1200,1200),'#151515');d=ImageDraw.Draw(canvas)
        for j,(_,r) in enumerate(hands.iloc[startrow:startrow+36].iterrows()):
            im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));b=r[CB].to_numpy(int);roi=Image.fromarray(im[b[1]:b[3],b[0]:b[2],::-1]);scale=min(195/roi.width,158/roi.height);roi=roi.resize((int(roi.width*scale),int(roi.height*scale)));x=(j%6)*200;y=(j//6)*200;canvas.paste(roi,(x,y+30));coords=(r[B].to_numpy(float)-np.r_[b[:2],b[:2]])*scale+np.array([x,y+30,x,y+30]);d.rectangle(tuple(coords),outline='#ff5555',width=2);d.text((x+2,y+2),str(startrow+j)+' '+r.frame_id[-15:],fill='white')
        canvas.save(A/f'audit/pseudo_hands_{page+1}.jpg')
    print('DONE',len(rec),time.time()-start,flush=True)
if __name__=='__main__':main()
