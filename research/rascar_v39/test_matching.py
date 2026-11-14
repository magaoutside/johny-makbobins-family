"""Behavior checks for overlap preservation and ambiguous duplicate decisions."""
from matching import *

def main():
    # Three valid .61 edges must beat two .99 edges: cardinality comes first.
    mat=np.array([[.99,.61,0],[0,.99,.61],[.61,0,0]])
    pairs,total=optimal_pairs(mat);assert len(pairs)==3 and abs(total-1.83)<1e-10
    # Distinct overlapping objects survive fusion; repeated observations fuse.
    boxes=[[0,0,100,100],[10,10,110,110]]
    data=[]
    for view in ['crop_a','crop_b']:
        for b in boxes:data.append(dict(record_id=view,confidence=.9,**dict(zip(B,b))))
    fused=fuse_observations(pd.DataFrame(data));assert len(fused)==2 and fused.views.eq(2).all()
    refs=pd.DataFrame([dict(instance=0,confidence=.9,views=2,**dict(zip(B,boxes[0])))])
    rows=pd.DataFrame([dict(id=f'q{k}',frame_id='synthetic',class_name='nude_hands',**dict(zip(B,boxes[0]))) for k in range(2)])
    evidence=assign_queries(rows,refs);assert evidence.duplicate_loser.sum()==1
    assert evidence.loc[evidence.duplicate_loser,'duplicate_margin'].eq(0).all()
    # A materially worse duplicate has a positive margin; an exact tie abstains.
    rows.loc[1,B]=[8,8,108,108];evidence=assign_queries(rows,refs)
    assert evidence.iloc[1].duplicate_loser and evidence.iloc[1].duplicate_margin>.03
    heads=pd.DataFrame([dict(zip(B,[10,10,20,20])),dict(zip(B,[110,10,120,20]))])
    proposals=pd.DataFrame([dict(record_id='a',confidence=.9,**dict(zip(B,[0,0,180,180]))),dict(record_id='b',confidence=.8,**dict(zip(B,[0,0,80,180]))),dict(record_id='c',confidence=.8,**dict(zip(B,[100,0,180,180])))])
    people=fuse_people_by_head(proposals,heads);assert len(people)==2 and people.views.eq(1).all()
    assert not any((people[B].to_numpy()==[0,0,180,180]).all(1))
    (A/'audit/matching_behavior_checks.json').write_text(json.dumps(dict(cardinality_before_iou=True,overlapping_instances_preserved=True,repeated_crop_views_fused=True,equal_duplicate_margin_zero=True,worse_duplicate_margin_positive=True,two_head_reference_rejected=True),indent=2))
    print('Six matching behavior checks passed.')

if __name__=='__main__':main()
