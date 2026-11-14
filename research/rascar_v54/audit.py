from common54 import *
from PIL import Image, ImageDraw, ImageFont
from prepare import errors


def main():
    rows = pd.read_csv(A/'inputs/records.csv'); q,t,h = inputs()
    originals = rows[rows.split.eq('oof')].set_index('id'); aug = rows[rows.split.eq('augmentation')]
    crop = pd.read_csv(ROOT/'rascar_v41/inputs/records.csv').set_index('id').loc[aug.parent_id,CB].to_numpy(float)
    boxes = aug[B].to_numpy(float)
    assert (boxes[:,:2] >= crop[:,:2]-1e-8).all() and (boxes[:,2:] <= crop[:,2:]+1e-8).all()
    assert (boxes[:,2:]-boxes[:,:2] >= 4).all()
    assert set(aug.parent_id) == set(q.id)
    assert set(aug.frame_id).isdisjoint(t.frame_id)
    assert rows[rows.split.eq('test')].error.isna().all()
    # Re-evaluate a deterministic stratified subset independently of saved labels.
    sample = aug.groupby(['family','class_name'],group_keys=False).sample(n=3, random_state=71)
    for _,r in sample.iterrows():
        group = q[q.frame_id.eq(r.frame_id)&q.class_name.eq(r.class_name)]
        gt = h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy()
        b = group[B].to_numpy().copy(); j=np.flatnonzero(group.id.eq(r.parent_id))[0]; b[j]=r[B].to_numpy(float)
        assert errors(b,gt)[j] == r.error
    static = pd.read_pickle(A/'features/static.pkl')
    assert not set(static)&{'sam_score','wevis_score','error','gt_iou','family','strength'}
    parent_index = rows[rows.family.eq('original')].reset_index().set_index('id')['index']
    changed = abs(static.loc[aug.index].to_numpy()-static.loc[parent_index.loc[aug.parent_id]].to_numpy()).max(1)>1e-7
    full = pd.read_pickle(A/'features/bank_fit25.pkl').drop(columns='class_name')
    full_changed = abs(full.loc[aug.index].to_numpy()-full.loc[parent_index.loc[aug.parent_id]].to_numpy()).max(1)>1e-7
    assert full_changed.all()
    report = dict(augmented_rows=len(aug), distinct_training_frames=aug.frame_id.nunique(),
                  independently_rechecked_labels=len(sample),
                  changed_target_vs_parent=int((aug.error.to_numpy()!=originals.loc[aug.parent_id,'error'].to_numpy()).sum()),
                  assignment_disagrees_with_independent_iou=int((aug.error.to_numpy()!=(aug.gt_iou<.6).astype(int).to_numpy()).sum()),
                  augmented_geometry_changed=int(changed.sum()),
                  augmented_full_features_changed=int(full_changed.sum()),
                  all_boxes_inside_frozen_context=True,test_frame_overlap=False,all_checks_passed=True)
    (A/'audit/integrity.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
    examples=[]
    for family,cls in [('shift','person'),('scale','person'),('edge','person'),('jitter','nude_hands'),('edge','nude_hands'),('toward_gt','person')]:
        eligible=aug[aug.family.eq(family)&aug.class_name.eq(cls)].copy()
        eligible=eligible[eligible.error.to_numpy()!=originals.loc[eligible.parent_id,'error'].to_numpy()]
        examples.append(eligible.iloc[len(eligible)//2])
    canvas=Image.new('RGB',(1200,800),'#20242a'); draw=ImageDraw.Draw(canvas)
    fontpath='/System/Library/Fonts/Supplemental/Arial.ttf'
    font=ImageFont.truetype(fontpath,16) if Path(fontpath).exists() else ImageFont.load_default()
    for n,r in enumerate(examples):
        x,y=(n%3)*400,(n//3)*400
        source=Image.open(D/f'{r.frame_id}.jpg').convert('RGB').resize((1280,720))
        orig=originals.loc[r.parent_id,B].to_numpy(float); box=r[B].to_numpy(float)
        lo=np.maximum(0,np.minimum(orig[:2],box[:2])-40); hi=np.minimum([1280,720],np.maximum(orig[2:],box[2:])+40)
        region=np.r_[lo,hi].astype(int); image=source.crop(tuple(region))
        ratio=min(380/image.width,330/image.height); size=(int(image.width*ratio),int(image.height*ratio)); image=image.resize(size)
        dx,dy=x+(400-size[0])//2,y+40+(330-size[1])//2; canvas.paste(image,(dx,dy))
        gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy()
        for bb,color in [(b,'#33ff77') for b in gt]+[(orig,'#ffe060'),(box,'#ff5555')]:
            bb=(bb-np.tile(region[:2],2))*ratio+np.tile([dx,dy],2)
            bb=np.maximum(bb,np.tile([dx,dy],2));bb=np.minimum(bb,np.tile([dx+size[0],dy+size[1]],2))
            if bb[2]>bb[0] and bb[3]>bb[1]:draw.rectangle(tuple(bb),outline=color,width=2)
        draw.text((x+10,y+8),f'{r.class_name}: {r.family}, error {int(originals.loc[r.parent_id,"error"])} -> {int(r.error)}',font=font,fill='white')
        draw.text((x+10,y+375),'GT green | original yellow | augmented red',font=font,fill='white')
    canvas.save(A/'audit/augmentation_examples.jpg',quality=94)


if __name__=='__main__':main()
