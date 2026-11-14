from common43 import *
from sklearn.metrics import average_precision_score as AP
from PIL import Image,ImageDraw,ImageFont
import argparse
p=argparse.ArgumentParser();p.add_argument('--name',default='pool_refinement_w010');p.add_argument('--local',action='store_true');p.add_argument('--root',default='rascar_v43');p.add_argument('--predictions');args=p.parse_args()
A=ROOT/args.root
q,te,h=inputs();bo,bt=baseline(q,te)
po=pd.read_csv(A/f'results/oof/{args.name}.csv').set_index('id').loc[q.id,'score'].to_numpy()
pt=pd.read_csv(A/f'results/submissions/{args.name}.csv').set_index('id').loc[te.id,'score'].to_numpy()
q=q.assign(old_score=bo,new_score=po,change=po-bo);te=te.assign(old_score=bt,new_score=pt,change=pt-bt)
f=pd.read_csv(A/'results'/(args.predictions or f'{"local_" if args.local else ""}pool_predictions.csv'))
if 'source' not in f:f['source']=f['split'].replace({'oof':'query'})
fo=f[f.source.eq('query')&f.day.ne(f.fit_day)].set_index('id');ft=f[f.source.eq('test')].set_index('id')
losers=[];pairs=[]
for i in np.flatnonzero(q.error.to_numpy()==1):
 for j in np.flatnonzero(q.error.to_numpy()==0):
    before=int(bo[i]<bo[j]);after=int(po[i]<po[j]);tie_before=bo[i]==bo[j];tie_after=po[i]==po[j]
    if before!=after or tie_before!=tie_after:pairs.append(dict(error_id=q.id.iloc[i],correct_id=q.id.iloc[j],before=before,after=after,tie_before=bool(tie_before),tie_after=bool(tie_after),status='repaired' if after<before else 'broken'))
pd.DataFrame(pairs).to_csv(A/f'audit/{args.name}_pairs.csv',index=False)
frames=q.frame_id.unique();rng=np.random.default_rng(4303);deltas=[]
for _ in range(2000):
 sampled=rng.choice(frames,len(frames),replace=True);ix=np.concatenate([np.flatnonzero(q.frame_id.eq(fid)) for fid in sampled]);deltas.append(AP(q.error.iloc[ix],po[ix])-AP(q.error.iloc[ix],bo[ix]))
loo=[]
for fid in frames:
 m=q.frame_id.ne(fid);loo.append(dict(excluded=fid,delta=AP(q.loc[m,'error'],po[m])-AP(q.loc[m,'error'],bo[m])))
pd.DataFrame(loo).to_csv(A/f'audit/{args.name}_leave_frame.csv',index=False)
st=dict(name=args.name,delta_ap=float(AP(q.error,po)-AP(q.error,bo)),bootstrap_95=np.quantile(deltas,[.025,.975]).tolist(),leave_frame_min=min(x['delta'] for x in loo),leave_frame_max=max(x['delta'] for x in loo),repaired_pairs=sum(x['status']=='repaired' for x in pairs),broken_pairs=sum(x['status']=='broken' for x in pairs),warning='Adaptive repeated OOF, not an independent confidence interval. No true test labels.')
(A/f'audit/{args.name}_stability.json').write_text(json.dumps(st,indent=2))
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',17);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
viewer=json.loads((ROOT/'rascar_test_viewer/data.js').read_text().removeprefix('window.RASCAR_DATA = ').rstrip(';\n'));rank={x['id']:x['rank'] for x in viewer['detections']}
def render(rows,split,name):
 for off in range(0,len(rows),6):
  sheet=Image.new('RGB',(1280,1200),'#171b23');dr=ImageDraw.Draw(sheet)
  for k,(_,r) in enumerate(rows.iloc[off:off+6].iterrows()):
   x=k%2*640;y=k//2*400;b=r[B].to_numpy(float);wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2
   source=fo if split=='oof' else ft
   refs=source.loc[[r.id]][['ref_'+v for v in B]].to_numpy(float) if r.id in source.index else np.empty((0,4));refs=refs[np.isfinite(refs).all(1)]
   gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy(float) if split=='oof' else np.empty((0,4))
   if len(gt):gt=gt[[iou(b,gt)[0].argmax()]]
   allb=np.concatenate([b[None],refs,gt]);lo=np.maximum(0,np.minimum(cc-np.maximum(wh*1.5,80)/2,allb[:,:2].min(0)-10));hi=np.minimum([1280,720],np.maximum(cc+np.maximum(wh*1.5,80)/2,allb[:,2:].max(0)+10))
   im=Image.open(D/f'{r.frame_id}.jpg').convert('RGB').resize((1280,720)).crop(tuple(np.r_[lo,hi].astype(int)));s=min(620/im.width,305/im.height);im=im.resize((round(im.width*s),round(im.height*s)),Image.Resampling.BICUBIC);sheet.paste(im,(x+10,y+90))
   scale=np.tile([im.width/(hi[0]-lo[0]),im.height/(hi[1]-lo[1])],2);offset=np.tile([x+10,y+90],2)
   for boxes,color in [(gt,'#5ff691'),(refs,'#67b4ff'),(b[None],'#ffda54')]:
    for bb in boxes:dr.rectangle(tuple((bb-np.tile(lo,2))*scale+offset),outline=color,width=2)
   label=f'GT error={int(r.error)}' if split=='oof' else f'test #{rank.get(r.id,"?")}'
   dr.text((x+10,y+4),f'{label} {r.class_name} {r.id}',font=font,fill='white');dr.text((x+10,y+29),f'V31 {r.old_score:.4f} -> {r.new_score:.4f}',font=font,fill='#dddddd');dr.text((x+10,y+54),'Yellow=query; blue=new ref; green=GT (OOF only)',font=small,fill='#b3c2cf')
  sheet.save(A/f'audit/{args.name}_{name}_{off//6:02d}.jpg',quality=94)
badids=set(x['error_id'] for x in pairs if x['status']=='broken')|set(x['correct_id'] for x in pairs if x['status']=='broken')
goodids=set(x['error_id'] for x in pairs if x['status']=='repaired')|set(x['correct_id'] for x in pairs if x['status']=='repaired')
render(q[q.id.isin(badids)],'oof','regressions');render(q[q.id.isin(goodids-badids)],'oof','repairs')
known=te[te.id.map(rank).isin([41,53,63,64,67,77,78,82,87,117])].copy();known['rank']=known.id.map(rank);render(known.sort_values('rank'),'test','discussed')
changed=te[abs(te.change)>1e-9].copy();changed['magnitude']=changed.change.abs();render(changed.nlargest(12,'magnitude'),'test','largest')
q.to_csv(A/f'audit/{args.name}_all_oof.csv',index=False);te.to_csv(A/f'audit/{args.name}_all_test.csv',index=False);print(json.dumps(st,indent=2))
