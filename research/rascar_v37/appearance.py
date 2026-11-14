"""Two frozen encoders, per-class supervised presence and exemplar comparison.

All fitted labels come from the fitting day's official manual boxes. Pool
reviews never enter fitting. Presence is distinct from bbox validity.
"""
from common37 import *
import os,time,cv2,torch,joblib
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('MPLCONFIGDIR',str(A/'models/mpl'))
from PIL import Image
from transformers import AutoModel,AutoImageProcessor,SiglipVisionModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(3);cv2.setNumThreads(1)

def records():
    manual=h.copy();manual['id']=['manual_'+str(i) for i in manual.index];manual['split']='manual'
    # Sample repeated bare-table crops evenly; preserve all non-table objects.
    tables=manual[manual.class_name.eq('table')].groupby(['day']).sample(n=30,random_state=3701)
    manual=pd.concat([manual[~manual.class_name.eq('table')],tables])
    rec=pd.concat([manual,q.assign(split='oof'),te.assign(split='test'),pd.read_csv(A/'inputs/blind_pool.csv')],ignore_index=True)
    if (A/'inputs/rival_pool.csv').exists():rec=pd.concat([rec,pd.read_csv(A/'inputs/rival_pool.csv')],ignore_index=True)
    rec=rec[['id','frame_id','day','class_name','split']+B];assert rec.id.is_unique
    rec.to_csv(A/'inputs/appearance_records.csv',index=False);return rec

def features(rec,kind):
    dest=A/f'features/{kind}_views.npy'
    if dest.exists():
        z=np.load(dest);assert len(z)==len(rec);return z
    model_path=ROOT/('rascar_v36/model_cache/siglip2-base-patch16-224' if kind=='siglip' else 'rascar_v31/model_cache/dinov2-small')
    model=(SiglipVisionModel if kind=='siglip' else AutoModel).from_pretrained(str(model_path),local_files_only=True).eval()
    if kind=='siglip':processor=AutoImageProcessor.from_pretrained(str(model_path),local_files_only=True,use_fast=False)
    result=[];start=time.time();mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
    for off in range(0,len(rec),8):
        batch=[]
        for _,r in rec.iloc[off:off+8].iterrows():
            raw=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720))[:,:,::-1]
            b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],4);cc=(b[:2]+b[2:])/2
            for scale in [1.05,1.5]:
                lo=np.maximum(0,np.floor(cc-scale*wh/2)).astype(int);hi=np.minimum([1280,720],np.ceil(cc+scale*wh/2)).astype(int)
                im=raw[lo[1]:hi[1],lo[0]:hi[0]]
                batch.append(Image.fromarray(im) if kind=='siglip' else cv2.resize(im,(224,224),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
        inputs=processor(images=batch,return_tensors='pt') if kind=='siglip' else {'pixel_values':(torch.from_numpy(np.stack(batch)).float()/255-mean)/std}
        with torch.inference_mode():
            out=model(**inputs);z=out.pooler_output if kind=='siglip' else out.last_hidden_state[:,0]
        result.append(torch.nn.functional.normalize(z,dim=1).numpy().reshape(-1,2,z.shape[1]))
        if off%160==0:print('FEATURES',kind,off,len(rec),round(time.time()-start,1),flush=True)
    z=np.concatenate(result);np.save(dest,z);return z

def targets(rec,cls):
    labels=np.full(len(rec),np.nan)
    for i,r in rec.iterrows():
        if r.split not in ['manual','oof']:continue
        boxes=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(cls)][B].to_numpy(float);b=r[B].to_numpy(float)
        if r.split=='manual' and r.class_name==cls:labels[i]=1;continue
        if not len(boxes):labels[i]=0;continue
        overlap=iou(b,boxes).ravel()
        cover=np.maximum(0,np.minimum(b[2:],boxes[:,2:])-np.maximum(b[:2],boxes[:,:2])).prod(1)/np.maximum((boxes[:,2:]-boxes[:,:2]).prod(1),1)
        if overlap.max()>=.3:labels[i]=1
        elif overlap.max()<.02 and cover.max()<.05:labels[i]=0
    return labels

def fit(rec,banks):
    output=[];manifest=[];labels={cls:targets(rec,cls) for cls in ['cup','tray','nude_hands']}
    for kind,xx in banks.items():
        x=xx.reshape(len(rec),-1)
        for cls,y in labels.items():
            for day in DAYS:
                train=rec.day.eq(day)&np.isfinite(y)&rec.split.isin(['manual','oof'])
                assert not set(rec.loc[train,'frame_id'])&set(rec.loc[rec.split.isin(['blind_pool','rival_pool','test']),'frame_id'])
                weights=1/rec.loc[train].groupby('frame_id').id.transform('size');weights*=train.sum()/weights.sum()
                model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=1500,random_state=3701))
                model.fit(x[train],y[train],logisticregression__sample_weight=weights)
                joblib.dump(model,A/f'models/{kind}_{cls}_fit{day[-2:]}.joblib')
                ev=rec.day.ne(day)&rec.split.isin(['manual','oof','blind_pool','rival_pool'])|rec.split.eq('test')
                prob=model.predict_proba(x[ev])[:,1]
                # Mean top-three cosine similarities; no probabilities claimed.
                xf=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
                pos=xf[train& (y==1)];neg=xf[train& (y==0)]
                simpos=np.sort(xf[ev]@pos.T,axis=1)[:,-min(3,len(pos)):].mean(1)
                simneg=np.sort(xf[ev]@neg.T,axis=1)[:,-min(3,len(neg)):].mean(1)
                d=rec.loc[ev].copy();d['target_class']=cls;d['fit_day']=day;d['encoder']=kind;d['p_present']=prob;d['semantic_target']=y[ev]
                d['positive_similarity']=simpos;d['negative_similarity']=simneg;d['exemplar_margin']=simpos-simneg;output.append(d)
                manifest.append(dict(encoder=kind,target_class=cls,fit_day=day,fit_frames=sorted(rec.loc[train,'frame_id'].unique()),n=int(train.sum()),positive=int(y[train].sum()),pool_in_fit=False,test_in_fit=False))
    pd.concat(output).to_csv(A/'features/appearance_predictions.csv',index=False)
    (A/'models/appearance_manifest.json').write_text(json.dumps(manifest,indent=2))

def evaluate():
    pred=pd.read_csv(A/'features/appearance_predictions.csv');results=[];pbank={};mbank={}
    for kind in ['siglip','dino','both']:
        pbank[kind]={};mbank[kind]={}
        for split,rows in ROWS.items():
            probs=np.ones(len(rows));margins=np.ones(len(rows))
            for i,r in rows.iterrows():
                pp=pred[pred.id.eq(r.id)&pred.target_class.eq(r.class_name)]
                if kind!='both':pp=pp[pp.encoder.eq(kind)]
                if len(pp):probs[i]=pp.p_present.max();margins[i]=pp.exemplar_margin.max()
            pbank[kind][split]=probs;mbank[kind][split]=margins
        for classes,label in [(['cup','tray'],'tableware'),(['nude_hands'],'hands')]:
            for cut in [.05,.15]:
                for nearest in [False,True]:
                    scores={}
                    for split,rows in ROWS.items():
                        p=pbank[kind][split];active=rows.class_name.isin(classes).to_numpy()&(p<cut)
                        if nearest:active&=mbank[kind][split]<0
                        scores[split]=BASE[split]+.25*np.maximum(1-p-BASE[split],0)*active
                    results.append(save(f'{kind}_{label}_absence{cut}_exemplar{int(nearest)}',scores,'Only class absence raises score; opposite-day fitting, both fit days must reject test. No geometry rescue.'))
    pd.DataFrame(results).to_csv(A/'results/appearance_experiments.csv',index=False)
    if not (A/'inputs/blind_review.json').exists():return
    lab=json.loads((A/'inputs/blind_review.json').read_text());pool=pd.read_csv(A/'inputs/blind_pool.csv')
    pool['presence']=np.nan
    for key,value in [('present',1),('absent',0)]:pool.loc[pool.review_index.isin(lab[key]),'presence']=value
    d=pred[pred.split.eq('blind_pool')&pred.class_name.eq(pred.target_class)].merge(pool[['id','review_index','presence']],on='id')
    d.to_csv(A/'audit/blind_pool_predictions.csv',index=False);summ=[]
    for (kind,cls),g in d[d.presence.notna()].groupby(['encoder','class_name']):
        summ.append(dict(encoder=kind,class_name=cls,n=len(g),present=int(g.presence.sum()),absent=int((g.presence==0).sum()),
            false_rejects_005=int(((g.presence==1)&(g.p_present<.05)).sum()),false_rejects_015=int(((g.presence==1)&(g.p_present<.15)).sum()),
            true_rejects_005=int(((g.presence==0)&(g.p_present<.05)).sum()),true_rejects_015=int(((g.presence==0)&(g.p_present<.15)).sum())))
    pd.DataFrame(summ).to_csv(A/'results/blind_pool_presence.csv',index=False);print(pd.DataFrame(summ).to_string(index=False))

if __name__=='__main__':
    rec=records();banks={kind:features(rec,kind) for kind in ['siglip','dino']};fit(rec,banks);evaluate()
