"""Freeze non-visual scoring evidence for a standalone CPU replay."""
from common18 import *
folder=A/'replay_inputs';folder.mkdir(exist_ok=True)
provenance={}
for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
    split='test' if test else 'oof'
    d=rows[['id','frame_id','class_name']+B+(['day','error'] if not test else [])].copy()
    d['base']=base;d['base16']=old.current if test else old.q.score.to_numpy()
    if not test:d['base_without_contradiction']=align(V17/'results/oof/release_balanced_without_v16_contradiction.csv',rows,'score')
    def add(path,prefix,cols=None):
        f=align(path,rows)
        for c in (cols or f.columns.tolist()):d[prefix+c]=f[c].to_numpy()
        provenance[str(path.relative_to(A.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
    add(A/f'person_body_ridge_{split}.csv','body_')
    add(V17/('seat_features_test_all.csv' if test else 'seat_features_oof.csv'),'seat_',['back','head_count','template_n','seat_distance'])
    path=V17/'results'/('submissions/submission_seat_union_w0.4.csv' if test else 'oof/seat_union_w0.4.csv')
    d['seat_delta04']=align(path,rows,'score')-d.base16
    add(A/f'hand_geometry_features_{split}.csv','geom_',['reference_iou','center_instances'])
    add(A/f'hand_geometry_catboost_{split}.csv','geom_',['prediction'])
    add(V16/f'semantic_tight_{split}.csv','tight_',['p_nude_hands','p_person'])
    add(V16/f'semantic_context_{split}.csv','context_',['p_person'])
    add(V16/f'calibrated_hand_{split}.csv','cal_',['assigned'])
    d['duplicate_veto']=duplicate_veto(rows,test)
    add(A/f'wrist_direction_{split}.csv','wrist_',['views','distal_fraction','proximal_fraction','projection','perp'])
    for n in ['v15','refined']:add(V17/f'tta_{n}_{split}.csv',n+'_',['assigned','overlap','views','turns','instances'])
    for n in ['v15','refined']:d[n+'_assigned']=boolean(d[n+'_assigned'])
    add(V16/f'wevis_features_{split}.csv','wevis_',['head_count','hand_iou'])
    add(V16/f'pose_features_{split}.csv','pose_',['pose_best_iou','pose_confirmed_views'])
    d.to_csv(folder/f'{split}.csv',index=False)
(folder/'provenance.json').write_text(json.dumps(provenance,indent=2))
