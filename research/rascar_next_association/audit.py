from common import *
from PIL import Image,ImageDraw,ImageFont
import argparse

def gallery(name,split):
    path=A/f'audit/{name}_changes.csv';d=pd.read_csv(path);d=d[d.split.eq(split)].copy();d=d.loc[d.delta.abs().sort_values(ascending=False).index]
    features=pd.read_csv(A/('features/query_features_local.csv' if 'bpj_local_' in name else 'features/query_features.csv')).set_index('id')
    qsrc=pd.concat([q,te],ignore_index=True).set_index('id');files=[]
    for off in range(0,len(d),9):
        canvas=Image.new('RGB',(1260,990),'#181818');draw=ImageDraw.Draw(canvas)
        for j,(_,r) in enumerate(d.iloc[off:off+9].iterrows()):
            src=qsrc.loc[r.id];f=features.loc[r.id];box=src[B].to_numpy(float);cc=(box[:2]+box[2:])/2;wh=np.maximum(box[2:]-box[:2], [80,65] if src.class_name=='nude_hands' else [200,200])*2.3
            cb=np.r_[np.maximum([0,0],np.floor(cc-wh/2)),np.minimum([1280,720],np.ceil(cc+wh/2))].astype(int)
            arr=image(src.frame_id)[cb[1]:cb[3],cb[0]:cb[2],::-1];pic=Image.fromarray(arr);scale=min(410/pic.width,255/pic.height);pic=pic.resize((round(pic.width*scale),round(pic.height*scale)));x=(j%3)*420;y=(j//3)*330
            canvas.paste(pic,(x,y+70));label=f'{r.id} {src.class_name}\n{r.old_score:.4f} -> {r.new_score:.4f}  delta {r.delta:+.4f}\n'
            label+=f'GT error={int(src.error)}  ' if split=='oof' else 'TEST (no GT)  '
            label+=f'refIoU={f.f_ref_iou:.3f} conf={f.f_part_confidence:.2f}'
            draw.text((x+3,y+3),label,fill='white')
            layers=[(box,'#ff4444')]
            if split=='oof':layers += [(g[B].to_numpy(float),'#44ff66') for _,g in h[h.frame_id.eq(src.frame_id)&h.class_name.eq(src.class_name)].iterrows()]
            layers += [(f[['ref_'+b for b in B]].to_numpy(float),'#40cfff')]
            for b,col in layers:
                if not np.isfinite(b).all():continue
                # Clip only for drawing; GT/evaluation coordinates stay unchanged.
                a=np.r_[np.maximum(b[:2],cb[:2]),np.minimum(b[2:],cb[2:])]
                if (a[2:]<=a[:2]).any():continue
                a=(a-np.r_[cb[:2],cb[:2]])*scale+np.array([x,y+70,x,y+70]);draw.rectangle(tuple(a),outline=col,width=2)
        p=A/f'audit/{name}_{split}_{off//9+1}.jpg';canvas.save(p,quality=90);files.append(str(p))
    return files

def stability(name):
    d=pd.read_csv(A/f'results/oof/{name}.csv').set_index('id').loc[q.id];s=d.score.to_numpy();base=BASE['oof'];y=q.error.to_numpy();err=np.flatnonzero(y);good=np.flatnonzero(1-y)
    before=base[err,None]>base[None,good];after=s[err,None]>s[None,good];pairs=[]
    for typ,m in [('improved',~before&after),('worsened',before&~after)]:
        for i,j in zip(*np.where(m)):pairs.append(dict(kind=typ,error_id=q.iloc[err[i]].id,correct_id=q.iloc[good[j]].id,error_frame=q.iloc[err[i]].frame_id,correct_frame=q.iloc[good[j]].frame_id))
    pd.DataFrame(pairs).to_csv(A/f'audit/{name}_pairs.csv',index=False)
    frames=q.frame_id.unique();leaves=[]
    for frame in frames:
        m=q.frame_id.ne(frame).to_numpy();leaves.append(dict(frame_id=frame,delta_ap=float(AP(y[m],s[m])-AP(y[m],base[m]))))
    pd.DataFrame(leaves).to_csv(A/f'audit/{name}_leave_frame_out.csv',index=False)
    rng=np.random.default_rng(4101);deltas=[];indices=pd.Index(frames).get_indexer(q.frame_id)
    for _ in range(3000):
        counts=np.bincount(rng.integers(len(frames),size=len(frames)),minlength=len(frames));weights=counts[indices]
        if np.unique(y[weights>0]).size<2:continue
        deltas.append(AP(y,s,sample_weight=weights)-AP(y,base,sample_weight=weights))
    changed=abs(s-base)>1e-10;f=pd.read_csv(A/('features/query_features_local.csv' if 'bpj_local_' in name else 'features/query_features.csv')).set_index('id')
    summary=dict(name=name,delta_ap=float(AP(y,s)-AP(y,base)),improved_pairs=int((~before&after).sum()),worsened_pairs=int((before&~after).sum()),frame_bootstrap_95=np.quantile(deltas,[.025,.975]).tolist(),leave_one_frame_delta_min=min(x['delta_ap'] for x in leaves),changed_oof=int(changed.sum()),correct_penalized=int((changed&(y==0)&(s>base)).sum()),errors_rescued_wrong_direction=int((changed&(y==1)&(s<base)).sum()),limits='Conditional diagnostic on repeatedly used OOF; no correction for adaptive selection. No independent newly annotated pool holdout.')
    (A/f'results/{name}_stability.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('name');args=p.parse_args();stability(args.name)
    for split in ['oof','test']:print('GALLERY',split,len(gallery(args.name,split)))

if __name__=='__main__':main()
