"""Fuse repeated crop observations, then match queries to separate instances."""
from common39 import *
from scipy.optimize import linear_sum_assignment
import argparse

def optimal_pairs(matrix):
    if not matrix.size:return [],0.
    valid=matrix>=.6;weight=np.where(valid,min(matrix.shape)+1+matrix,0.)
    ii,jj=linear_sum_assignment(weight,maximize=True)
    pairs=[(int(i),int(j)) for i,j in zip(ii,jj) if valid[i,j]]
    return pairs,float(sum(matrix[i,j] for i,j in pairs))

def fuse_observations(group):
    """Complete-link geometric clusters; never merge two slots of the same view.

    IoU .8 exceeds all distinct same-class overlaps in official manual data
    (max .571). The threshold is fixed, not optimized against query error labels.
    """
    group=group.sort_values('confidence',ascending=False);clusters=[]
    for _,r in group.iterrows():
        box=r[B].to_numpy(float);view=r.record_id;eligible=[]
        for j,c in enumerate(clusters):
            if view in c['views']:continue
            sim=iou(box,np.array(c['boxes'])).ravel()
            if sim.min()>=.8:eligible.append((float(sim.mean()),j))
        if eligible:
            j=max(eligible)[1];c=clusters[j];c['boxes'].append(box);c['weights'].append(float(r.confidence));c['views'].add(view)
        else:clusters.append(dict(boxes=[box],weights=[float(r.confidence)],views={view}))
    out=[]
    for j,c in enumerate(clusters):
        bb=np.average(c['boxes'],axis=0,weights=c['weights'])
        out.append(dict(instance=j,confidence=float(np.mean(c['weights'])),views=len(c['views']),**dict(zip(B,bb))))
    return pd.DataFrame(out,columns=['instance','confidence','views']+B)

def fuse_people_by_head(group,head_rows):
    """Assign person proposals to an independently detected head, not a query ID.
    One proposal per crop/head votes; coordinate medians resist a broad crop.
    """
    hh=head_rows[B].to_numpy(float);centers=(hh[:,:2]+hh[:,2:])/2;accepted=[]
    for _,r in group.iterrows():
        box=r[B].to_numpy(float);inside=((centers>=box[:2])&(centers<=box[2:])).all(1)
        if inside.sum()!=1:continue
        j=int(np.flatnonzero(inside)[0])
        if (centers[j,1]-box[1])/max(box[3]-box[1],1)>.5:continue
        accepted.append(dict(head=j,record_id=r.record_id,confidence=r.confidence,**dict(zip(B,box))))
    out=[]
    if accepted:
        d=pd.DataFrame(accepted).sort_values('confidence',ascending=False).drop_duplicates(['head','record_id'])
        for j,g in d.groupby('head'):
            box=np.median(g[B].to_numpy(),axis=0);out.append(dict(instance=int(j),confidence=float(g.confidence.mean()),views=len(g),**dict(zip(B,box))))
    return pd.DataFrame(out,columns=['instance','confidence','views']+B)

def assign_queries(rows,references):
    mat=iou(rows[B].to_numpy(),references[B].to_numpy());n=len(rows);assigned=np.full(n,-1)
    pairs,total=optimal_pairs(mat)
    for i,j in pairs:assigned[i]=j
    out=[]
    for i,(_,r) in enumerate(rows.iterrows()):
        best=int(mat[i].argmax()) if mat.shape[1] else -1;maximum=float(mat[i,best]) if best>=0 else np.nan
        chosen=int(assigned[i]) if assigned[i]>=0 else best
        duplicate=bool(best>=0 and maximum>=.6 and assigned[i]<0);margin=0.
        if duplicate:
            alternatives=[]
            for j in np.flatnonzero(mat[i]>=.6):
                rest=np.delete(np.delete(mat,i,axis=0),j,axis=1);alt,value=optimal_pairs(rest)
                if len(alt)+1==len(pairs):alternatives.append(value+mat[i,j])
            margin=max(0.,total-max(alternatives)) if alternatives else 1.
        z=dict(id=r.id,frame_id=r.frame_id,class_name=r.class_name,assigned=assigned[i]>=0,duplicate_loser=duplicate,duplicate_margin=margin,max_overlap=maximum,reference_count=len(references))
        if chosen>=0:
            ref=references.iloc[chosen];z.update(overlap=float(mat[i,chosen]),reference_confidence=float(ref.confidence),instance=int(ref.instance),**{f'ref_{b}':ref[b] for b in B})
        else:z.update(overlap=np.nan,reference_confidence=0.,instance=-1,**{f'ref_{b}':np.nan for b in B})
        out.append(z)
    return pd.DataFrame(out)

def collect_evidence(model_run='set',run_name='set',identity=False,full_model_run=None):
    evidence=[];instances=[];heads=pd.read_csv(ROOT/'rascar_v16/wevis_heads_references.csv') if identity else None
    (A/f'inputs/{run_name}_matching_protocol.json').write_text(json.dumps(dict(model_run=model_run,full_model_run=full_model_run,run_name=run_name,identity=identity,source_class_views_only=identity,confidence=.5,fusion_iou=.8,instance_seed_iou=.7,duplicate_margin=.03),indent=2))
    for fit in list(DAYS)+(['all'] if full_model_run else []):
        for seed in [3901,3902,3903]:
            name=f'{full_model_run}_fitall' if fit=='all' else f'{model_run}_fit{fit[-2:]}'
            pred=pd.read_csv(A/f'features/{name}_seed{seed}_boxes.csv')
            pred=pred[pred.confidence.ge(.5)&~pred.crop_edge.astype(bool)]
            splits=[('test',te)] if fit=='all' else [('oof',q[q.day.ne(fit)])]+([] if full_model_run else [('test',te)])
            for split,rows in splits:
                d=pred[pred.record_id.isin(rows.id)]
                for (frame,cls),group in rows.groupby(['frame_id','class_name']):
                    raw=d[d.frame_id.eq(frame)&d.class_name.eq(cls)]
                    if identity:raw=raw[raw.record_id.isin(group.id)]
                    refs=fuse_people_by_head(raw,heads[heads.frame_id.eq(frame)]) if identity and cls=='person' else fuse_observations(raw)
                    z=assign_queries(group,refs);z['fit_day']=fit;z['seed']=seed;z['split']=split
                    # Diagnose old union references using independently predicted instances.
                    keys=['oof'] if split=='oof' else (['test_fit25','test_fit26'] if fit=='all' else ['test_fit'+fit[-2:]])
                    old=[pd.read_csv(ROOT/f'rascar_v31/features/linear_mask_cut0.5_{key}_evidence.csv').set_index('id') for key in keys];merged=[]
                    centers=(refs[B[:2]].to_numpy()+refs[B[2:]].to_numpy())/2
                    for ident in z.id:
                        counts=[]
                        for bank in old:
                            bb=bank.loc[ident,['ref_'+b for b in B]].to_numpy(float)
                            counts.append(int(((centers>=bb[:2])&(centers<=bb[2:])).all(1).sum()) if np.isfinite(bb).all() else 0)
                        merged.append(min(counts)>=2)
                    z['old_reference_multi_instance']=merged;evidence.append(z)
                    if len(refs):instances.append(refs.assign(frame_id=frame,class_name=cls,fit_day=fit,seed=seed,split=split))
    pd.concat(evidence,ignore_index=True).to_csv(A/f'features/{run_name}_matching_evidence.csv',index=False)
    pd.concat(instances,ignore_index=True).to_csv(A/f'features/{run_name}_frame_instances.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model-run',default='set');parser.add_argument('--run-name',default='set');parser.add_argument('--identity',action='store_true');parser.add_argument('--full-model-run');args=parser.parse_args();collect_evidence(args.model_run,args.run_name,args.identity,args.full_model_run)
