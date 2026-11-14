"""Exact ranking-loss audit of frozen V31. Official OOF truth only."""
from common40 import *
from collections import Counter

def main():
    out=A/'audit/baseline';out.mkdir(parents=True,exist_ok=True)
    d=q.copy();y=d.error.to_numpy();s=d.score.to_numpy();p=int(y.sum())
    ev=pd.read_csv(ROOT/'rascar_v31/features/linear_mask_cut0.5_oof_evidence.csv').set_index('id').loc[d.id].reset_index()
    details=[]
    for i,r in d.iterrows():
        above=s>=s[i];pos=int(y[above].sum());neg=int((1-y[above]).sum())
        details.append(dict(id=r.id,precision_at_score=pos/(pos+neg),negatives_above_or_tied=neg if y[i] else 0,
            positives_below_or_tied=int(((s<=s[i])&(y==1)).sum()) if not y[i] else 0,
            ap_loss=(1-pos/(pos+neg))/p if y[i] else 0.))
    d=d.merge(pd.DataFrame(details),on='id',validate='one_to_one')
    for c in ['overlap','confidence','references']+['ref_'+b for b in B]:d['dense_'+c]=ev[c].to_numpy()
    geo=[]
    for _,r in d.iterrows():
        gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)]
        ov=iou(r[B],gt[B].to_numpy()).ravel();j=int(ov.argmax()) if len(ov) else None
        best=float(ov[j]) if j is not None else 0.;bb=gt.iloc[j][B].to_numpy(float) if j is not None else np.full(4,np.nan)
        ref=r[['dense_ref_'+b for b in B]].to_numpy(float)
        refgt=float(iou(ref,gt[B]).max()) if len(gt) and np.isfinite(ref).all() else np.nan
        kind='correct' if not r.error else ('one_to_one_conflict' if best>=.6 else 'near_boundary_0.5_to_0.6' if best>=.5 else 'geometry_overlap_0.1_to_0.5' if best>=.1 else 'no_strong_same_class_match')
        geo.append(dict(id=r.id,gt_max_iou=best,automatic_type=kind,dense_reference_gt_iou=refgt,**dict(zip(['gt_'+b for b in B],bb))))
    d=d.merge(pd.DataFrame(geo),on='id',validate='one_to_one')
    assert abs(d.ap_loss.sum()-(1-AP(y,s)))<1e-12
    d.sort_values('ap_loss',ascending=False).to_csv(out/'all_198_official_queries.csv',index=False)
    positives=d[d.error.eq(1)&d.ap_loss.gt(0)].sort_values('ap_loss',ascending=False)
    negatives=d[d.error.eq(0)&d.positives_below_or_tied.gt(0)].sort_values('positives_below_or_tied',ascending=False)
    positives.to_csv(out/'under_ranked_errors.csv',index=False);negatives.to_csv(out/'over_ranked_correct.csv',index=False)
    intrusion=d[d.error.eq(0)].copy()
    intrusion['ap_loss_intrusions']=[sum(1/(p*int((s>=v).sum())) for v in d[d.error.eq(1)&d.score.le(value)].score) for value in intrusion.score]
    assert abs(intrusion.ap_loss_intrusions.sum()-d.ap_loss.sum())<1e-12
    intrusion.sort_values('ap_loss_intrusions',ascending=False).to_csv(out/'correct_intrusion_loss.csv',index=False)
    classes=[]
    for cls,g in d.groupby('class_name'):
        classes.append(dict(class_name=cls,queries=len(g),frames=g.frame_id.nunique(),errors=int(g.error.sum()),correct=int((1-g.error).sum()),class_ap=AP(g.error,g.score),ap_loss=g.ap_loss.sum(),share_ap_loss=g.ap_loss.sum()/d.ap_loss.sum(),under_ranked_errors=int((g.ap_loss>0).sum()),over_ranked_correct=int((g.positives_below_or_tied>0).sum())))
    pd.DataFrame(classes).to_csv(out/'by_class.csv',index=False)
    pairs=[]
    for _,a in d[d.error.eq(1)].iterrows():
        for _,b in d[d.error.eq(0)&d.score.ge(a.score)].iterrows():
            pairs.append(dict(error_id=a.id,correct_id=b.id,error_class=a.class_name,correct_class=b.class_name,error_day=a.day,correct_day=b.day,same_day=a.day==b.day,same_frame=a.frame_id==b.frame_id,tie=a.score==b.score,error_score=a.score,correct_score=b.score))
    pairs=pd.DataFrame(pairs);pairs.to_csv(out/'misordered_pairs.csv',index=False)
    pd.crosstab(pairs.error_class,pairs.correct_class).to_csv(out/'misordered_class_pairs.csv')
    dist=[]
    for split,rows in [('oof',q),('test',te)]:
        for cls,g in rows.groupby('class_name'):
            widths=g.x2-g.x1;heights=g.y2-g.y1
            dist.append(dict(split=split,class_name=cls,n=len(g),fraction=len(g)/len(rows),frames=g.frame_id.nunique(),median_width=widths.median(),median_height=heights.median(),median_confidence=g.confidence.median()))
    pd.DataFrame(dist).to_csv(out/'observable_distribution.csv',index=False)
    for field in ['day','frame_id']:
        d.groupby(field).agg(queries=('id','size'),ap_loss=('ap_loss','sum')).sort_values('ap_loss',ascending=False).to_csv(out/f'loss_by_{field}.csv')
    stat=dict(ap=float(AP(y,s)),queries=len(d),frames=d.frame_id.nunique(),positive_errors=p,correct=len(d)-p,under_ranked_errors=len(positives),over_ranked_correct=len(negatives),misordered_pairs=len(pairs),ties=int(pairs.tie.sum()),cross_day_pairs=int((~pairs.same_day).sum()),same_frame_pairs=int(pairs.same_frame.sum()),top5_error_share_loss=float(positives.ap_loss.head(5).sum()/d.ap_loss.sum()),top10_error_share_loss=float(positives.ap_loss.head(10).sum()/d.ap_loss.sum()),unique_frames_in_ranking_errors=pd.concat([positives,negatives]).frame_id.nunique())
    (out/'summary.json').write_text(json.dumps(stat,indent=2));print(json.dumps(stat,indent=2));print(pd.DataFrame(classes).to_string(index=False))
    print('UNDER RANKED');print(positives[['id','class_name','error','score','gt_max_iou','ap_loss','negatives_above_or_tied']].to_string(index=False))
    print('OVER RANKED');print(negatives[['id','class_name','score','gt_max_iou','positives_below_or_tied']].to_string(index=False))
    gallery(pd.concat([positives,negatives],ignore_index=True),out/'official_rank_errors')

def gallery(rows,path):
    from PIL import Image,ImageDraw
    for start in range(0,len(rows),9):
        batch=rows.iloc[start:start+9];canvas=Image.new('RGB',(1500,1410),'#181818');dr=ImageDraw.Draw(canvas)
        for j,(_,r) in enumerate(batch.iterrows()):
            b=r[B].to_numpy(float);gb=r[['gt_'+v for v in B]].to_numpy(float);bb=np.stack([b,gb]) if np.isfinite(gb).all() and r.gt_max_iou>0 else b[None]
            low=bb[:,:2].min(0);high=bb[:,2:].max(0);wh=np.maximum(high-low,32);center=(low+high)/2
            lo=np.maximum(0,np.floor(center-.9*wh)).astype(int);hi=np.minimum([1280,720],np.ceil(center+.9*wh)).astype(int)
            im=Image.open(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg').resize((1280,720)).crop((*lo,*hi));scale=min(495/im.width,375/im.height);im=im.resize((int(im.width*scale),int(im.height*scale)))
            x=j%3*500;yy=j//3*470;canvas.paste(im,(x,yy+85))
            def box(v,color):
                if not np.isfinite(v).all():return
                v=(np.asarray(v)-np.r_[lo,lo])*scale+[x,yy+85,x,yy+85];v[[0,2]]=np.clip(v[[0,2]],x,x+im.width-1);v[[1,3]]=np.clip(v[[1,3]],yy+85,yy+85+im.height-1)
                if v[2]>v[0] and v[3]>v[1]:dr.rectangle(tuple(v),outline=color,width=2)
            for _,g in h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)].iterrows():box(g[B].to_numpy(float),'#20ff70')
            box(r[['dense_ref_'+v for v in B]].to_numpy(float),'#4caeff');box(b,'#ff4040')
            for line,txt in enumerate([f'{start+j}: {r.class_name} {r.id}',f'error={r.error} score={r.score:.4f} GT IoU={r.gt_max_iou:.4f}',f'loss={r.ap_loss:.6f} negabove={r.negatives_above_or_tied} errbelow={r.positives_below_or_tied}',r.frame_id]):dr.text((x+3,yy+2+18*line),txt,fill='white')
        canvas.save(Path(str(path)+f'_{start//9+1}.jpg'))

if __name__=='__main__':main()
