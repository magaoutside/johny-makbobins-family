"""Feature bank. Query/error labels never enter image features or embeddings."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,cv2
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v41'))
from common import inputs,B,CB,CL,D
sys.path.pop(0)
cv2.setNumThreads(1)
BASE13=['sam_score','wevis_score','best_iou','cover_query','cover_ref','log_width_ratio','log_height_ratio','ref_confidence','zoom_iou','hand_iou','head_count','pose_best_iou']
EXTRA10=['top_offset','bottom_offset','left_offset','right_offset','no_reference','hand_query_coverage','hand_instances','head_near_outside','pose_body_outside','pose_confirmed_views']
def stats(im):
    im=cv2.resize(im,(96,96));rgb=im[:,:,::-1]/255.;hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)/np.array([180,255,255]);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);z=gray/255.;out={}
    for prefix,v in [('rgb',rgb),('hsv',hsv)]:
        for j in range(3):
            for name,fn in [('mean',np.mean),('std',np.std),('q10',lambda x:np.quantile(x,.1)),('q90',lambda x:np.quantile(x,.9))]:out[f'{prefix}{j}_{name}']=float(fn(v[:,:,j]))
    p=np.bincount((gray//16).flatten(),minlength=16)/gray.size
    out.update(entropy=float(-(p*np.log(p+1e-9)).sum()),bright=float((z>.94).mean()),dark=float((z<.06).mean()),log_laplacian=float(np.log1p(cv2.Laplacian(gray,cv2.CV_32F).var())))
    edges=cv2.Canny(gray,60,150)/255.;yy,xx=np.mgrid[:96,:96];border=(xx<12)|(xx>=84)|(yy<12)|(yy>=84);out['edges']=edges.mean();out['border_edges']=edges[border].mean();out['center_edges']=edges[~border].mean()
    ycrcb=cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb);skin=(ycrcb[:,:,1]>133)&(ycrcb[:,:,1]<173)&(ycrcb[:,:,2]>77)&(ycrcb[:,:,2]<127);out['heuristic_skin_fraction']=skin.mean()
    gx=cv2.Sobel(z,cv2.CV_32F,1,0,ksize=3) if z.dtype==np.float32 else cv2.Sobel(z.astype(np.float32),cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(z.astype(np.float32),cv2.CV_32F,0,1,ksize=3);mag,angle=cv2.cartToPolar(gx,gy);bins=np.floor(angle/np.pi*8).astype(int)%8
    for yi in range(2):
        for xi in range(2):
            m=(yy//48==yi)&(xx//48==xi);hist=np.bincount(bins[m],weights=mag[m],minlength=8);hist/=np.linalg.norm(hist)+1e-6
            for k,v in enumerate(hist):out[f'gradient_{yi}{xi}_{k}']=v
    for k,v in enumerate(p):out[f'gray_bin{k}']=v
    return out
def main():
    q,t,h=inputs();rows=pd.concat([q.assign(split='oof'),t.assign(split='test')],ignore_index=True);assert rows.id.is_unique;rows.to_csv(A/'inputs/rows.csv',index=False)
    blocks=[];manifest=[]
    for stem,train,test in [
      ('raw',ROOT/'rascar_v16/inputs/v15/previous/rascar_v14/stage1/raw_frozen_train_features.csv',ROOT/'rascar_v16/inputs/v15/previous/rascar_v14/stage1/raw_frozen_test_features.csv'),
      ('wevis',ROOT/'rascar_v16/wevis_features_oof.csv',ROOT/'rascar_v16/wevis_features_test.csv'),
      ('pose',ROOT/'rascar_v16/pose_features_oof.csv',ROOT/'rascar_v16/pose_features_test.csv')]:
        d=pd.concat([pd.read_csv(train),pd.read_csv(test)],ignore_index=True);assert d.id.is_unique and set(d.id)==set(rows.id);blocks.append(d.set_index('id').loc[rows.id]);manifest.extend([dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [train,test]])
    static=pd.concat(blocks,axis=1);assert static.columns.is_unique
    classcols=[x for x in static if x.startswith('class_')];static=static.drop(columns=classcols)
    groups={'base13':BASE13,'base23':BASE13+EXTRA10,'frozen_all':[c for c in static if c not in ['cx','cy','w','h','log_area','log_aspect','confidence']],'position':['cx','cy','w','h','log_area','log_aspect','confidence']}
    owl=pd.read_csv(ROOT/'rascar_v50/features/features.csv').set_index('id').loc[rows.id];cols=[c for c in owl if c not in ['frame_id','day','split','class_name','error','detector_confidence'] and not c.startswith('class_')];groups['owl']=['owl_'+c for c in cols]
    for c in cols:static['owl_'+c]=owl[c]
    rec=pd.read_csv(ROOT/'rascar_v41/inputs/records.csv');lookup=rec.reset_index().set_index('id')['index'];assert set(rows.id)<=set(lookup.index)
    simrec=pd.read_csv(ROOT/'rascar_v41/inputs/clip_text_records.csv');idx=simrec.reset_index().set_index('id').loc[rows.id,'index'];sim=np.load(ROOT/'rascar_v41/features/clip_text_similarities.npy')[idx]
    textnames=['person','hands','cup','tray','elbow','sleeve','phone','fabric','bottle','plate','table','chair','bag','dog'];groups['clip_text']=['clip_text_'+c for c in textnames]
    for j,c in enumerate(groups['clip_text']):static[c]=sim[:,j]
    ci=rows.class_name.map(CL.index).to_numpy();static['clip_class_margin']=sim[np.arange(len(rows)),ci]-np.max(np.where(np.arange(14)[None,:]==ci[:,None],-10,sim),axis=1);groups['clip_text'].append('clip_class_margin')
    cache={};visual=[];recidx=rec.set_index('id')
    for n,r in rows.iterrows():
        if r.frame_id not in cache:cache[r.frame_id]=cv2.resize(cv2.imread(str(D/f'{r.frame_id}.jpg')),(1280,720))
        im=cache[r.frame_id];b=np.clip(np.r_[np.floor(r[B][:2]),np.ceil(r[B][2:])],[0,0,1,1],[1279,719,1280,720]).astype(int);cb=recidx.loc[r.id,CB].to_numpy(int);a=stats(im[b[1]:b[3],b[0]:b[2]]);c=stats(im[cb[1]:cb[3],cb[0]:cb[2]]);d={'id':r.id}
        d.update({'pixel_box_'+k:v for k,v in a.items()});d.update({'pixel_context_'+k:v for k,v in c.items()});d.update({'pixel_difference_'+k:a[k]-c[k] for k in a});visual.append(d)
    pixels=pd.DataFrame(visual).set_index('id').loc[rows.id];static=pd.concat([static,pixels],axis=1);groups['pixels']=pixels.columns.tolist()
    for key in ['clip','vit_b','vit_l']:
        src=ROOT/f'rascar_v41/features/{key}_frozen.npy';f=np.load(src,mmap_mode='r');dim=(f.shape[1]-13)//3;x=np.asarray(f[lookup.loc[rows.id]],np.float32)[:,:dim*3];np.save(A/f'features/{key}.npy',x)
        # Exact GT boxes (variant 0) supply independent semantic prototypes in fit day only.
        proto=rec[rec.split.eq('synthetic')&rec.id.str.startswith('synth_')&rec.id.str.endswith('_0')].copy();ix=proto.index.to_numpy();np.save(A/f'features/{key}_prototypes.npy',np.asarray(f[ix],np.float32)[:,:dim*3]);proto[['id','frame_id','day','class_name']].to_csv(A/'inputs/prototypes.csv',index=False)
        print('CACHED',key,x.shape,flush=True)
    dense_rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');dl=dense_rec.reset_index().set_index('id')['index'];rr=dense_rec.set_index('id').loc[rows.id];b=rr[B].to_numpy();cb=rr[CB].to_numpy();yy,xx=np.mgrid[:32,:32];xx=cb[:,0,None,None]+(xx[None]+.5)*(cb[:,2]-cb[:,0])[:,None,None]/32;yy=cb[:,1,None,None]+(yy[None]+.5)*(cb[:,3]-cb[:,1])[:,None,None]/32;m=(xx>=b[:,0,None,None])&(xx<=b[:,2,None,None])&(yy>=b[:,1,None,None])&(yy<=b[:,3,None,None]);assert m.any((1,2)).all()
    for key,path in [('dino_small',ROOT/'rascar_v31/features/dense_grids.npy'),('siglip2',ROOT/'rascar_v46/features/siglip2_grids.npy'),('vit_b448',ROOT/'rascar_v46/features/vit_b_grids.npy'),('vit_l448',ROOT/'rascar_v46/features/vit_l_grids.npy')]:
        f=np.load(path,mmap_mode='r');out=[]
        for j,i in enumerate(dl.loc[rows.id]):
            z=np.asarray(f[i],np.float32);inside=z[:,m[j]].mean(1);outside=z[:,~m[j]].mean(1);out.append(np.r_[z.mean((1,2)),inside,inside-outside])
        np.save(A/f'features/{key}.npy',np.asarray(out,np.float32));print('POOLED',key,np.shape(out),flush=True)
    static['class_name']=rows.class_name.to_numpy();assert np.isfinite(static.drop(columns='class_name').to_numpy(float)).all();static.reset_index().to_csv(A/'features/static.csv',index=False)
    (A/'inputs/groups.json').write_text(json.dumps(groups,indent=2));(A/'inputs/source_manifest.json').write_text(json.dumps(manifest,indent=2))
    (A/'inputs/protocol.json').write_text(json.dumps(dict(target='error of one declared-class bbox, NOT image junk',baseline='V31, public AP 0.96239 reported by user',official_queries=len(q),test_queries=len(t),class_feature='categorical string',validation='Train day25 -> day26 and reverse. PCA/scalers/prototypes fitted within training day. Learned dense heads use saved V48 purged inner predictions for fit-day and outer opposite-day predictions for evaluation.',embeddings=['CLIP ViT-B/32','DINOv2 small 448','DINOv2 base 224/448','DINOv2 large 224/448','SigLIP2 base 512'],other_models=['SAM and WeVis frozen scores from supplied V17 source tables','WeVis head/hand boxes','multi-view pose','frozen OWLv2'],limitations='Repeated development OOF. Broad feature search entails selection bias. Test and subjective test statuses never used as targets; no claim to test AP.',blend='(baseline + alpha * p_error)/(1+alpha)',alpha=[.05,.1,.2,.5],initial_models=['CatBoost depth2 250 iterations seed23','CatBoost depth4 400 iterations seed23','LogReg C0.1'],avoid='No early stopping against held day. No baseline score, ID/day/frame/error/GT boxes as classifier features.'),indent=2))
    print('DONE',static.shape,flush=True)
if __name__=='__main__':main()
