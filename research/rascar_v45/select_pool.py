"""Select fresh difficult AND random machine queries before seeing manual labels."""
from common45 import *
from PIL import Image,ImageDraw,ImageFont
def main():
    seen=set();sources=[]
    for p in ROOT.glob('rascar*/inputs/*.csv'):
        if A in p.parents:continue
        try:
            d=pd.read_csv(p,usecols=['frame_id']);s=set(d.frame_id.dropna())
            if len(s)<1500:seen|=s;sources.append(str(p.relative_to(ROOT)))
        except (ValueError,pd.errors.EmptyDataError):pass
    for folder in ['rascar_v44/pool_review','rascar_v37','rascar_v38/inputs']:
        for p in (ROOT/folder).glob('*.csv'):
            try:seen|=set(pd.read_csv(p,usecols=['frame_id']).frame_id.dropna())
            except (ValueError,pd.errors.EmptyDataError):pass
    pool=pd.read_csv(D/'machine_predictions.csv').rename(columns={'detection_id':'id'});pool=pool[pool.class_name.isin(CL)].copy()
    for fname in ['train_stream_predictions.csv','test_stream_predictions.csv']:
        seen|=set(pd.read_csv(D/fname,usecols=['frame_id']).frame_id)
    def secs(fid):return pd.Timestamp(fid.split('_')[-2]+' '+fid.split('_')[-1]).timestamp()
    pool['seconds']=pool.frame_id.map(secs);pool['day']=pool.frame_id.map(lambda s:s.split('_')[-2]);pool=pool[~pool.frame_id.isin(seen)].copy()
    # Exclude 120 seconds around previously recorded frames, not just exact images.
    oldtime=np.array(sorted(secs(x) for x in seen))
    dt=np.abs(pool.seconds.to_numpy()[:,None]-oldtime[None]).min(1);pool=pool[dt>120].copy()
    pool['neighbor_iou']=0.
    for _,g in pool.groupby(['frame_id','class_name']):
        ov=iou(g[B],g[B]);np.fill_diagonal(ov,0);pool.loc[g.index,'neighbor_iou']=ov.max(1)
    wh=pool[['x2','y2']].to_numpy()-pool[['x1','y1']].to_numpy();pool['aspect']=np.maximum(wh[:,0]/wh[:,1],wh[:,1]/wh[:,0]);pool['area']=wh.prod(1)
    rng=np.random.default_rng(4501);pool['random_order']=rng.random(len(pool))
    # A scene/time block cannot supply many visually redundant examples.
    selected=[];chosen=[]
    for cls,n in [('person',24),('nude_hands',24),('cup',12),('tray',12)]:
        for day in ['20261025','20261026']:
            g=pool[pool.class_name.eq(cls)&pool.day.eq(day)].copy()
            # Half randomized; half stressed for neighbors, low confidence, elongated geometry.
            g['difficulty']=g.neighbor_iou*2+(1-g.confidence)+.15*np.minimum(g.aspect,4)
            for mode in ['random','difficult']:
                order=g.sort_values('random_order' if mode=='random' else 'difficulty',ascending=mode=='random')
                got=0
                for _,r in order.iterrows():
                    if chosen and min(abs(r.seconds-np.array(chosen)))<300:continue
                    selected.append(dict(r,selection=mode));chosen.append(r.seconds);got+=1
                    if got>=n//4:break
    df=pd.DataFrame(selected).reset_index(drop=True);df['review_index']=df.index
    # Reserve before review; rejected cases are not replaced to make holdout easy.
    df['partition']=np.where(df.index%3==0,'reserved','development')
    rois=[]
    for _,r in df.iterrows():
        bb=r[B].to_numpy(float);cc=(bb[:2]+bb[2:])/2;size=np.maximum((bb[2:]-bb[:2])*1.7,100)
        cb=np.r_[np.maximum(0,np.floor(cc-size/2)),np.minimum([1280,720],np.ceil(cc+size/2))];rois.append(cb)
    df[CB]=np.array(rois);df.to_csv(A/'pool_review/candidates.csv',index=False)
    (A/'pool_review/selection.json').write_text(json.dumps(dict(seed=4501,count=len(df),reserved=int(df.partition.eq('reserved').sum()),seen_frames=len(seen),excluded_time_radius_seconds=120,selected_spacing_seconds=300,source_ledgers=sources,labels_seen_at_selection=False,test_labels=False),indent=2))
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
    for off in range(0,len(df),8):
        sheet=Image.new('RGB',(1400,1400),'#171b23');dr=ImageDraw.Draw(sheet)
        for j,(_,r) in enumerate(df.iloc[off:off+8].iterrows()):
            x=j%2*700;y=j//2*350;cb=r[CB].to_numpy(float);im=Image.open(D/f'{r.frame_id}.jpg').convert('RGB');native=np.array([im.width/1280,im.height/720]);im=im.crop(tuple(cb*np.tile(native,2)))
            s=min(660/im.width,270/im.height);im=im.resize((int(im.width*s),int(im.height*s)),Image.Resampling.NEAREST);sheet.paste(im,(x+10,y+65))
            scale=np.tile([im.width/(cb[2]-cb[0]),im.height/(cb[3]-cb[1])],2);box=(r[B].to_numpy(float)-np.tile(cb[:2],2))*scale+np.tile([x+10,y+65],2)
            dr.rectangle(tuple(box),outline='#ffdd00',width=2)
            dr.text((x+10,y+5),f"#{r.review_index} {r.class_name} conf={r.confidence:.2f} {r.partition}",font=font,fill='white');dr.text((x+10,y+27),f"{r.frame_id[-15:]} {r.selection} crop {cb.astype(int).tolist()}",font=font,fill='white')
        sheet.save(A/f'pool_review/candidates_{off//8:02}.jpg')
    print(df.groupby(['class_name','partition']).size().to_string())
if __name__=='__main__':main()
