"""Cross-day ranking and held-day localization checks for new vote heads."""
from common36 import *

def collect(boxes,rows,decoder,confidence,identity=False):
 heads=pd.read_csv(ROOT/'rascar_v16/wevis_heads_references.csv') if identity else None
 groups={k:d for k,d in boxes[boxes.decoder.eq(decoder)&boxes.confidence.ge(confidence)].groupby(['record_id','class_name'])};out=[]
 for _,r in rows.iterrows():
  ref=groups.get((r.id,r.class_name),boxes.iloc[:0]);raw_n=len(ref);head_n=-1
  if identity and r.class_name=='person':
   hb=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy(float);hc=(hb[:,:2]+hb[:,2:])/2;b=r[B].to_numpy(float)
   inside=((hc>=b[:2])&(hc<=b[2:])).all(1);head_n=int(inside.sum())
   if head_n!=1:ref=ref.iloc[:0]
   elif len(ref):
    rb=ref[B].to_numpy(float);same=((hc[inside][0]>=rb[:,:2])&(hc[inside][0]<=rb[:,2:])).all(1)
    count=((hc[None]>=rb[:,None,:2])&(hc[None]<=rb[:,None,2:])).all(2).sum(1)
    # Mere inclusion is insufficient: a neighboring torso can contain the
    # query person's head near its bottom. For an upright/seated person, the
    # head must be in the upper half of its proposed reference rectangle.
    head_y_fraction=(hc[inside][0,1]-rb[:,1])/np.maximum(rb[:,3]-rb[:,1],1e-6)
    ref=ref.loc[same&(count==1)&(head_y_fraction<=.5)]
  entry=dict(id=r.id,valid=False,overlap=np.nan,confidence=0.,vote_coherence=0.,vote_count=0,references=len(ref),raw_references=raw_n,query_heads=head_n,**{'ref_'+b:np.nan for b in B})
  if len(ref):
   ov=iou(r[B],ref[B]).ravel();j=int(ov.argmax());rr=ref.iloc[j]
   entry.update(valid=bool(ov[j]>=.05),overlap=float(ov[j]),confidence=float(rr.confidence),vote_coherence=float(rr.vote_coherence),vote_count=int(rr.vote_count),**{'ref_'+b:float(rr[b]) for b in B})
  out.append(entry)
 return pd.DataFrame(out)

def guard(rows,split):
 z=align(ROOT/f'rascar_v22/replay_inputs/{split}.csv',rows)
 sem=align(ROOT/f'rascar_v16/semantic_both_{split}.csv',rows)
 st=align(ROOT/f'rascar_v21/object_structure_{split}.csv',rows)
 yes=lambda s:s.astype(str).str.lower().isin(['true','1','1.0']).to_numpy()
 ok=np.ones(len(rows),bool)
 for cls in CL:
  m=rows.class_name.eq(cls).to_numpy()
  if cls=='nude_hands':
   veto=z.independent_hand_centers.ge(2).to_numpy()|yes(z.old_duplicate_veto)|yes(z.zoom_duplicate_loser)|(np.minimum(z.quality_logreg,z.quality_svm)>=.75)|z.structural_hand.gt(z.baseline+1e-9).to_numpy()
   ok[m]=((~veto)&z.semantic_hand.ge(.8).to_numpy())[m]
  elif cls in ['cup','tray']:ok[m]=((sem['p_'+cls]>=.8)&st.distinct_instances.le(1)&~yes(st.ambiguous_duplicate)).to_numpy()[m]
 return ok

def combine(kind,decoder,confidence):
 identity=kind.endswith('_identity');source=kind.replace('_identity','')
 fs={};controls=[];rec=pd.read_csv(V31/'inputs/dense_records.csv')
 for split,rows in ROWS.items():
  parts=[]
  for fit in DAYS:
   box=pd.read_csv(A/f'features/{source}_fit{fit[-2:]}_boxes.csv')
   if 'view_scale' in box:box=box[box.view_scale.eq(2.4)]
   rr=rows[rows.day.ne(fit)] if split=='oof' else rows
   d=collect(box,rr,decoder,confidence,identity);d['fit_day']=fit;parts.append(d)
   d.to_csv(A/f'features/{kind}_{decoder}_c{confidence}_{split}_fit{fit[-2:]}.csv',index=False)
   if split=='oof':
    manual=rec[rec.split.eq('manual')&rec.day.ne(fit)&rec.class_name.isin(CL)]
    md=collect(box,manual,decoder,confidence,identity).merge(manual[['id','origin_id','class_name']],on='id')
    for cls in CL:
     oo=md[md.class_name.eq(cls)].groupby('origin_id').overlap.mean().fillna(0)
     controls.append(dict(kind=kind,decoder=decoder,confidence=confidence,held_day='2026-10-26' if fit.endswith('25') else '2026-10-25',class_name=cls,n=len(oo),mean_iou=float(oo.mean()),recall06=float(oo.ge(.6).mean())))
  if split=='oof':
   d=pd.concat(parts).set_index('id').loc[rows.id];target=np.clip((.85-d.overlap.to_numpy())/.5,0,1);valid=d.valid.to_numpy(bool)
  else:
   parts=[d.set_index('id').loc[rows.id] for d in parts];raw=np.stack([np.clip((.85-d.overlap.to_numpy())/.5,0,1) for d in parts]);delta=raw-BASE[split]
   valid=np.logical_and.reduce([d.valid.to_numpy(bool) for d in parts])&(np.all(delta>=0,0)|np.all(delta<=0,0))
   rb=[d[['ref_'+b for b in B]].to_numpy(float) for d in parts]
   valid &= np.array([iou(a,b)[0,0]>=.6 for a,b in zip(*rb)])
   target=np.where(np.all(delta>=0,0),np.min(raw,0),np.max(raw,0))
  delta=np.nan_to_num(target-BASE[split],nan=0.)
  # A corroborated geometric ref cannot silently erase old semantic/duplicate vetoes.
  valid &= (delta>=0)|guard(rows,split)
  fs[split]=(valid,delta)
 return fs,controls

def main():
 results=[save('v31_frozen',BASE,'Confirmed Public .96239')];controls=[]
 for kind in ['dense','center']:
  for decoder in ['vote','peak']:
   for confidence in [.35,.55]:
    fs,cc=combine(kind,decoder,confidence);controls+=cc
    for family,classes in [('person',['person']),('hands',['nude_hands']),('tableware',['cup','tray']),('all',CL)]:
     for weight in [.1,.25]:
      scores={}
      for split,rows in ROWS.items():
       valid,delta=fs[split];active=valid & rows.class_name.isin(classes).to_numpy()
       scores[split]=np.clip(BASE[split]+weight*delta*active,0,1)
      results.append(save(f'{kind}_{decoder}_c{confidence}_{family}_w{weight}',scores,'Class-specific box votes, train-day manual+empty; conservative cross-day agreement on test; previous semantic/duplicate vetoes retained.'))
 pd.DataFrame(results).sort_values('pooled_ap',ascending=False).to_csv(A/'results/vote_experiments.csv',index=False)
 pd.DataFrame(controls).to_csv(A/'results/manual_localization.csv',index=False)

if __name__=='__main__':main()
