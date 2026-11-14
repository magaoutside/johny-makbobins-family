"""A previously unused TRAIN-pool hand challenge set; never test annotations.

Select before looking at model outputs. One target per ten-minute spatial
bucket, excluding all prior model training/control/review frame IDs. Labels
are added only after image inspection; uncertain boundary cases remain unset.
"""
from experiments import *
from PIL import Image,ImageDraw
import cv2

def main():
    p=pd.read_csv(ROOT/'rascar_data_full/machine_predictions.csv')
    p['day']=pd.to_datetime(p.frame_id.str[-15:-7],format='%Y%m%d').dt.strftime('%Y-%m-%d')
    excluded=set(q.frame_id)
    for name in ['rascar_v22/inputs/manual_boxes.csv','rascar_v33/inputs/pseudo_records.csv',
                 'rascar_v33/inputs/rare_records.csv','rascar_v33/inputs/rare_hand_review_candidates.csv',
                 'rascar_v32/inputs/pool_control_records.csv']:
        excluded |= set(pd.read_csv(ROOT/name).frame_id)
    z=p[p.day.isin(DAYS)&p.class_name.eq('nude_hands')&~p.frame_id.isin(excluded)].copy()
    z['cx']=(z.x1+z.x2)/2;z['cy']=(z.y1+z.y2)/2
    z['min_side']=np.minimum(z.x2-z.x1,z.y2-z.y1)
    z=z[z.min_side.ge(14)&z.min_side.le(85)]
    stamp=pd.to_datetime(z.frame_id.str[-15:],format='%Y%m%d_%H%M%S').astype('int64')//(600*10**9)
    z['stratum']=np.where(z.confidence<.65,'lower_conf',np.where((z.cy>400)|(z.cx>1000),'periphery','standard'))
    z['bucket']=z.day+'_'+stamp.astype(str)+'_'+(z.cx//120).astype(int).astype(str)+'_'+(z.cy//100).astype(int).astype(str)
    rng=np.random.default_rng(3407);z['draw']=rng.random(len(z))
    z=z.sort_values('draw').drop_duplicates('bucket').drop_duplicates('frame_id')
    picked=[]
    for day in DAYS:
        for group in ['lower_conf','periphery','standard']:
            picked.append(z[z.day.eq(day)&z.stratum.eq(group)].head(6))
    sel=pd.concat(picked,ignore_index=True);sel=sel.rename(columns={'detection_id':'id'})
    sel['review_index']=np.arange(len(sel));sel['split']='fresh_pool'
    cb=[]
    for _,r in sel.iterrows():
        b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2
        cb.append(np.r_[np.maximum(0,np.floor(cc-1.6*wh)),np.minimum([1280,720],np.ceil(cc+1.6*wh))])
    sel[['crop_x1','crop_y1','crop_x2','crop_y2']]=np.stack(cb)
    assert not set(sel.frame_id)&excluded and set(sel.day)<=set(DAYS)
    sel.to_csv(A/'inputs/fresh_pool_candidates.csv',index=False)
    (A/'inputs/fresh_pool_selection.json').write_text(json.dumps(dict(seed=3407,queries=len(sel),
        frames=int(sel.frame_id.nunique()),excluded_frames=len(excluded),test_frames_used=False,
        selection='6 random per day/stratum after temporal-spatial deduplication; no model output or visual label used in sampling',
        limitation='Stratified diagnostic set, not representative AP estimate. No official labels. Query frames unused in previous fitting/review; nearby context may have appeared in prior train-day data.'),indent=2))
    for off in range(0,len(sel),12):
        rr=sel.iloc[off:off+12];canvas=Image.new('RGB',(1400,1050),'#181818');d=ImageDraw.Draw(canvas)
        for j,(_,r) in enumerate(rr.iterrows()):
            im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720))
            c=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(int)
            roi=Image.fromarray(im[c[1]:c[3],c[0]:c[2],::-1]);scale=min(340/roi.width,280/roi.height)
            roi=roi.resize((int(roi.width*scale),int(roi.height*scale)));x=j%4*350;y=j//4*350
            canvas.paste(roi,(x,y+55));box=(r[B].to_numpy(float)-np.r_[c[:2],c[:2]])*scale+np.array([x,y+55,x,y+55])
            d.rectangle(tuple(box),outline='#ff4444',width=2)
            d.text((x+4,y+4),f'{r.review_index}  {r.day[-2:]}  {r.stratum}',fill='white')
            d.text((x+4,y+20),f'{r.frame_id[-6:]}  native {r.min_side/2:.0f}px',fill='white')
        canvas.save(A/f'audit/fresh_pool_{off//12+1}.jpg')
    print(sel.groupby(['day','stratum']).size().to_string(),flush=True)

if __name__=='__main__':main()
