"""A targeted tray geometry review, separated before inspection."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent;D=ROOT/'rascar_data_full'
for p in ['pool_review','inputs','features','models','logs','results','audit']:(A/p).mkdir(parents=True,exist_ok=True)
B=['x1','y1','x2','y2'];CB=['crop_x1','crop_y1','crop_x2','crop_y2']
def main():
    candidates=pd.read_csv(D/'machine_predictions.csv');candidates=candidates[candidates.class_name.eq('tray')].copy();candidates['time']=pd.to_datetime(candidates.frame_id.str[-15:],format='%Y%m%d_%H%M%S');candidates['day']=candidates.time.dt.strftime('%Y-%m-%d');candidates['area']=(candidates.x2-candidates.x1)*(candidates.y2-candidates.y1)
    sources=['rascar_v33/inputs/pseudo_records.csv','rascar_v34/inputs/fresh_pool_candidates.csv','rascar_v37/inputs/blind_pool.csv','rascar_v44/pool_review/reviews.csv','rascar_v45/pool_review/reviews.csv','rascar_v41/pool_expansion/visual_review.csv']
    excluded=set()
    for s in sources:
        p=ROOT/s
        if p.exists():
            f=pd.read_csv(p)
            if 'frame_id' in f:excluded.update(f.frame_id)
    for name in ['train_stream_predictions.csv','test_stream_predictions.csv','human_boxes_train.csv']:excluded.update(pd.read_csv(D/name).frame_id)
    et=pd.to_datetime(pd.Series(sorted(excluded)).str[-15:],format='%Y%m%d_%H%M%S').astype('int64').to_numpy()/1e9
    times=candidates.time.astype('int64').to_numpy()/1e9;eligible=(np.abs(times[:,None]-et[None])>120).all(1);candidates=candidates[eligible].copy();rng=np.random.default_rng(4901);selected=[];used=[]
    for day,g in candidates.groupby('day'):
        # Half uniform candidates, half unusual area/confidence; all time-separated.
        random=g.iloc[rng.permutation(len(g))];hard=g.assign(priority=abs(np.log(g.area)-np.log(g.area).median())+.5*(1-g.confidence)).sort_values('priority',ascending=False)
        n=0
        for group,label,budget in [(random,'random',9),(hard,'unusual',18)]:
            for _,r in group.iterrows():
                t=r.time.timestamp()
                if used and min(abs(t-z) for z in used)<300:continue
                d=r.to_dict();d['selection']=label;d['role']='reserved' if n%3==0 else 'development';d['review_index']=len(selected)
                b=r[B].to_numpy(float);cc=(b[:2]+b[2:])/2;wh=np.maximum(b[2:]-b[:2],50)*2.2;lo=np.maximum(0,np.floor(cc-wh/2));hi=np.minimum([1280,720],np.ceil(cc+wh/2));d.update(dict(zip(CB,np.r_[lo,hi])));selected.append(d);used.append(t);n+=1
                if n>=budget:break
            if n>=18:break
    strict_count=len(selected)
    # Training-only supplement may be near own-day train context or previously seen.
    # Never claim this supplement as an independent validation set. Outer day split
    # still keeps the entire validation day/context out of fitting.
    more=pd.read_csv(D/'machine_predictions.csv');more=more[more.class_name.eq('tray')].copy();more['time']=pd.to_datetime(more.frame_id.str[-15:],format='%Y%m%d_%H%M%S');more['day']=more.time.dt.strftime('%Y-%m-%d');more['area']=(more.x2-more.x1)*(more.y2-more.y1)
    forbidden=set(pd.read_csv(D/'human_boxes_train.csv').frame_id)|set(pd.read_csv(D/'test_stream_predictions.csv').frame_id)
    more=more[~more.frame_id.isin(forbidden)]
    for day,g in more.groupby('day'):
        n=0
        for _,r in g.iloc[rng.permutation(len(g))].iterrows():
            t=r.time.timestamp()
            if used and min(abs(t-z) for z in used)<300:continue
            d=r.to_dict();d.update(selection='development_supplement',role='development',review_index=len(selected));b=r[B].to_numpy(float);cc=(b[:2]+b[2:])/2;wh=np.maximum(b[2:]-b[:2],50)*2.2;lo=np.maximum(0,np.floor(cc-wh/2));hi=np.minimum([1280,720],np.ceil(cc+wh/2));d.update(dict(zip(CB,np.r_[lo,hi])));selected.append(d);used.append(t);n+=1
            if n>=8:break
    out=pd.DataFrame(selected);out.to_csv(A/'pool_review/candidates.csv',index=False)
    (A/'inputs/selection.json').write_text(json.dumps(dict(planned_strict=36,strict_selected=strict_count,selected=len(out),role_counts=out.role.value_counts().to_dict(),excluded_frames=len(excluded),exclusion_sources=sources,purge_seconds=120,minimum_separation_seconds=300,seed=4901,reserved_before_inspection=True,development_supplement='May overlap past seen frames or own-day training context; excludes official manual frames, test stream and reserved neighborhoods. Used only with outer-day exclusion.',caveat='Excluded recorded prior reviews for strict subset, not proof of never appearing in a historical unlogged screenshot. No pool label is official ground truth.'),indent=2))
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',11)
    for off in range(0,len(out),6):
        sheet=Image.new('RGB',(1320,1080),'#161c25');dr=ImageDraw.Draw(sheet)
        for k,(_,r) in enumerate(out.iloc[off:off+6].iterrows()):
            x=k%3*440;y=k//3*540;cb=r[CB].to_numpy(int);im=Image.open(D/f'{r.frame_id}.jpg').convert('RGB').resize((1280,720)).crop(tuple(cb));s=min(400/im.width,420/im.height);sz=(round(im.width*s),round(im.height*s));im=im.resize(sz,Image.Resampling.BICUBIC);sheet.paste(im,(x+30,y+95));dr.text((x+8,y+5),f'#{int(r.review_index)} {r.role} {r.day[-2:]} conf={r.confidence}',font=font,fill='white');dr.text((x+8,y+28),r.frame_id[-15:],font=font,fill='white');dr.text((x+8,y+51),f'query {np.round(r[B].to_numpy(float),1).tolist()}',font=small,fill='white');dr.text((x+8,y+70),f'crop origin {cb[:2].tolist()}, local ticks',font=small,fill='white')
            bb=(r[B].to_numpy(float)-np.tile(cb[:2],2))*s+np.tile([x+30,y+95],2);dr.rectangle(bb.tolist(),outline='#ffdd44',width=2)
            for xx in range(0,cb[2]-cb[0],20):
                px=x+30+int(xx*s);dr.line((px,y+91,px,y+95),fill='white');dr.text((px,y+80),str(xx),font=small,fill='#7bddff')
            for yy in range(0,cb[3]-cb[1],20):
                py=y+95+int(yy*s);dr.line((x+26,py,x+30,py),fill='white');dr.text((x+1,py-5),str(yy),font=small,fill='#7bddff')
        sheet.save(A/f'pool_review/candidates_{off//6:02d}.jpg',quality=95)
    print(out[['review_index','day','role','selection','frame_id']].to_string(index=False))
if __name__=='__main__':main()
