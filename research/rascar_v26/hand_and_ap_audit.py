"""Read-only re-audit of V25 hand changes and the submitted person probe."""
from common26 import *
from PIL import Image, ImageDraw, ImageFont
import math
V25=ROOT/'rascar_v25'
FONT=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)

def hand_audit():
    new=align(V25/'submission_v25_pose_conservative_research.csv',te,'score')
    z=audit.copy();z['baseline']=current;z['candidate']=new;z['change']=new-current
    changed=z[abs(z.change)>1e-9].sort_values('review_no').copy()
    models=['rtmw_padwide','wilor_detector_padwide','rtmwx_bgr_padwide','rtmwx_rgb_padwide']
    feats={m:(align(V25/f'features/{m}_test_{DAYS[0]}.csv',te)+align(V25/f'features/{m}_test_{DAYS[1]}.csv',te))/2 for m in models}
    for m,f in feats.items():
        changed[m+'_iou']=f.loc[changed.id,'best_iou'].to_numpy()
    # Audit the full duplicate groups, not just the row receiving a rescue.
    groups=[]
    for r in changed.itertuples():
        peers=z[z.frame_id.eq(r.frame_id)&z.class_name.eq('nude_hands')]
        overlap=iou(np.array([getattr(r,k) for k in B]),peers[B])[0]
        peers=peers[overlap>.35]
        for pr in peers.itertuples():
            rec=dict(target_review=r.review_no,review_no=pr.review_no,id=pr.id,baseline=pr.baseline,candidate=pr.candidate)
            rec.update({m+'_iou':float(f.loc[pr.id,'best_iou']) for m,f in feats.items()});groups.append(rec)
    changed.to_csv(A/'audit/hand_changes.csv',index=False)
    pd.DataFrame(groups).to_csv(A/'audit/hand_duplicate_groups.csv',index=False)
    for start in range(0,len(changed),4):
        page=Image.new('RGB',(1280,900),'white');dr=ImageDraw.Draw(page)
        for k,r in enumerate(list(changed.itertuples())[start:start+4]):
            ox=(k%2)*640;oy=(k//2)*450
            b=np.array([getattr(r,c) for c in B]);cc=(b[:2]+b[2:])/2;size=145
            cr=np.r_[cc-size/2,cc+size/2]
            im=Image.open(ROOT/f'rascar_v16/test_images/{r.frame_id}.jpg').convert('RGB')
            pic=im.crop(tuple(cr)).resize((420,350));page.paste(pic,(ox,oy+70))
            def box(bb,color,label=''):
                xy=(np.array(bb)-np.tile(cr[:2],2))*np.tile([420/size,350/size],2)+[ox,oy+70,ox,oy+70]
                dr.rectangle(tuple(xy),outline=color,width=2)
                if label:dr.text((xy[0],xy[1]-17),label,font=FONT,fill=color,stroke_width=1,stroke_fill='black')
            dr.text((ox+5,oy+4),f'#{r.review_no}: {r.baseline:.6f} -> {r.candidate:.6f}',font=FONT,fill='black')
            peers=z[z.frame_id.eq(r.frame_id)&z.class_name.eq('nude_hands')]
            for pr in peers.itertuples():
                pb=np.array([getattr(pr,c) for c in B])
                if iou(b,pb)[0,0]>.35:box(pb,'yellow' if pr.id==r.id else '#ff6030',str(pr.review_no))
            for j,(m,f) in enumerate(feats.items()):
                row=f.loc[r.id];color=['#00FFFF','#00FF00','#FF77FF','#55AAFF'][j]
                box(row[['ref_'+c for c in B]],color)
                dr.text((ox+425,oy+80+j*55),m.replace('_padwide',''),font=FONT,fill='black')
                dr.text((ox+425,oy+100+j*55),f'IoU {row.best_iou:.3f}',font=FONT,fill='black')
            dr.text((ox+5,oy+430),r.visual_status+' / '+r.issue,font=FONT,fill='black')
        page.save(A/f'audit/hand_evidence_{start//4}.jpg',quality=95)
    print(changed[['review_no','baseline','candidate']+[m+'_iou' for m in models]].to_string(index=False))
    print(pd.DataFrame(groups).to_string(index=False))

def ap_audit():
    new=align(V25/'submission_v25_person_hidden_legs_test.csv',te,'score')
    diff=abs(new-current)>1e-9
    d0=current[:,None]-current[None,:];d1=new[:,None]-new[None,:]
    upper=np.triu(np.ones(d0.shape,dtype=bool),1);flipped=(d0*d1<0)&upper
    ia,ib=np.where(flipped);participants=np.unique(np.r_[ia,ib])
    # Explicit counterexample: all crossed endpoints are errors. All public
    # rows are retained and ranking changes substantially, but AP stays equal.
    y=(np.arange(len(te))%5==0).astype(int);y[participants]=1
    example=dict(description='Artificial labels, not inferred competition labels; all 362 rows public',positive_count=int(y.sum()),before=float(AP(y,current)),after=float(AP(y,new)))
    assert abs(example['before']-example['after'])<1e-12
    label=audit.visual_status.map({'plausible_correct':0,'clear_error':1,'likely_error':1}).to_numpy()
    pairs=[]
    for i,j in zip(ia,ib):
        kind='unknown' if np.isnan(label[i]) or np.isnan(label[j]) else 'same_visual_label' if label[i]==label[j] else 'visually_favourable' if (new[i]-new[j])*(label[i]-label[j])>0 else 'visually_adverse'
        pairs.append(dict(id_a=te.id.iloc[i],id_b=te.id.iloc[j],review_a=int(audit.review_no.iloc[i]),review_b=int(audit.review_no.iloc[j]),visual_type=kind))
    result=dict(reported_lb_v22=.95610,reported_lb_probe=.95610,report_precision_decimals=5,
       changed_rows=int(diff.sum()),changed_frames=int(te.loc[diff,'frame_id'].nunique()),all_deltas_positive=bool((new[diff]>current[diff]).all()),
       reversed_pairs=len(ia),participating_rows=len(participants),old_tied_pairs=int(((d0==0)&upper).sum()),new_tied_pairs=int(((d1==0)&upper).sum()),
       visual_pair_counts=pd.Series([p['visual_type'] for p in pairs]).value_counts().to_dict(),artificial_counterexample=example,
       conclusion='Identical displayed AP does not identify the public/private split or validate hidden-leg labels.',
       conditional_statement='If all 12 changed rows truly are positives and exact AP is unchanged, no changed public positive crosses a public negative. The labels and exact score are unknown.',
       rounding='Five-decimal equality is compatible with a nonzero difference below 1e-5; mixed true labels may offset gains and losses.',
       validation='Frozen visual torso-only judgments are hypotheses; official OOF penalizes some such boxes as correct.')
    (A/'results/person_lb_analysis.json').write_text(json.dumps(result,indent=2));pd.DataFrame(pairs).to_csv(A/'audit/person_lb_crossed_pairs.csv',index=False)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    hand_audit();ap_audit()
