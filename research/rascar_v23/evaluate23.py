"""Cross-day evaluation of explicit CV rules and compact per-class models."""
from common23 import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier
import joblib

PIXEL=['color_strength','brightness','glare','contrast','skin_mean','skin_p90','skin_component_iou','skin_component_area',
       'skin_component_solidity','skin_component_aspect','edge_density','contour_round','contour_rect','contour_solid',
       'fg_references','fg_inside','fg_bottom','fg_top','fg_left','fg_right']
fo=pd.concat([pd.read_csv(A/f'features/held_{fit}.csv') for fit in DAYS]).set_index('id').loc[q.id]
testfold=[align(A/f'features/test_{fit}.csv',te) for fit in DAYS];ft=(testfold[0]+testfold[1])/2
fs={'oof':fo,'test':ft}

def rule_masks(f,rows):
    hand=rows.class_name.eq('nude_hands').to_numpy();person=rows.class_name.eq('person').to_numpy()
    tray=rows.class_name.eq('tray').to_numpy();cup=rows.class_name.eq('cup').to_numpy()
    # Evidence needs at least two pose views; all color-based decisions require color.
    distal=(f.wrist_views>=2)&(f.wrist_distance<.85)&(f.wrist_projection>-.15)&(f.wrist_projection<.8)&(f.wrist_perp<.45)&(f.wrist_distal_fraction>=.75)
    masks={
      'body_visible_overflow':person&(f.head_count==1)&(f.body_pose_views>=2)&(f.body_point_iou_q75<.55),
      'body_seat_range':person&(f.head_count==1)&(f.seat_support>=4)&(f.seat_iou_max<.6)&(f.seat_bottom_min_gap>.12)&(f.seat_height_min_ratio<.8)&(f.edge==0),
      'body_seat_foreground':person&(f.head_count==1)&(f.seat_support>=3)&(f.seat_iou_max<.67)&(f.seat_bottom_min_gap>.12)&(f.fg_bottom>.25)&(f.edge==0),
      'hand_distal_skin':hand&distal&(f.color_strength>5)&(f.skin_component_iou>.55)&(f.skin_component_area>.12)&(f.skin_component_area<1.5),
      'hand_distal_shape':hand&distal&(f.hand_size_to_forearm>.3)&(f.hand_size_to_forearm<1.1)&(abs(f.size_width_z)<1.7)&(abs(f.size_height_z)<1.7),
      'hand_proximal_forearm':hand&(f.wrist_views>=2)&(f.wrist_distance<1.5)&(f.wrist_projection<-.4)&(f.wrist_perp<.35),
      'tray_without_support':tray&(f.table_coverage<.15)&(f.wrist_distance>2.5)&(f.person_coverage<.15),
      'tray_round_plate':tray&(f.contour_round>.8)&(f.contour_rect<.87)&(f.tray_best_iou<.45)&(f.table_coverage>.65),
      'cup_carried':cup&(f.wrist_views>=2)&(f.wrist_distance<1.3)&(f.cup_best_iou>=.5)&(f.cup_centers==1)
    }
    return {k:np.asarray(v,bool) for k,v in masks.items()}

def veto(split,rows):
    # Preserve existing duplicate/merged-hand protection and independent semantic veto.
    z=align(V22/f'replay_inputs/{split}.csv',rows)
    truth=lambda s:s.astype(str).str.lower().isin(['true','1','1.0']).to_numpy()
    return ((z.independent_hand_centers.to_numpy()>=2)|truth(z.old_duplicate_veto)|truth(z.zoom_duplicate_loser)|
        (np.minimum(z.quality_logreg,z.quality_svm).to_numpy()>=.75)|
        (z.structural_hand.to_numpy()>z.baseline.to_numpy()+1e-9)), z

def rules():
    masks={split:rule_masks(f,rows) for split,f,rows in [('oof',fo,q),('test',ft,te)]}
    for name in masks['oof']:
        rescue=name in ['hand_distal_skin','hand_distal_shape','cup_carried']
        for weight in [.2,.4]:
            scores=[]
            for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
                gate=masks[split][name].copy();f=fs[split]
                if rescue:
                    vv,zz=veto(split,rows);gate &= ~vv
                    if name.startswith('hand'):gate &= zz.semantic_hand.ge(.8).to_numpy()
                    else:
                        sem=align(ROOT/f'rascar_v16/semantic_both_{split}.csv',rows,'p_cup')
                        st=align(ROOT/f'rascar_v21/object_structure_{split}.csv',rows)
                        gate &= (sem>=.8)&st.distinct_instances.le(1).to_numpy()&~st.ambiguous_duplicate.astype(bool).to_numpy()
                    target=np.full(len(rows),.35)
                    if name=='hand_distal_skin':target=np.clip((.85-f.skin_component_iou.to_numpy())/.5,.1,.6)
                    s=base-weight*np.maximum(base-target,0)*gate
                else:s=base+weight*(1-base)*gate
                pd.DataFrame(dict(id=rows.id,gate=gate,score=s)).to_csv(A/f'features/{name}_{weight}_{split}.csv',index=False)
                scores.append(s)
            register(f'rule_{name}_w{weight}',*scores,dict(type='explicit_CV_rule',rule=name,weight=weight,no_test_id_rules=True))

def models():
    variants=[('relations','logreg'),('pixels','catboost'),('combined','logreg'),('combined','catboost')]
    for kind,algorithm in variants:
        po=np.zeros(len(q));testprob=[];standalone=[]
        for fit,held in zip(DAYS,DAYS[::-1]):
            tr=pd.read_csv(A/f'inputs/train_{fit}.csv');x=pd.read_csv(A/f'features/train_{fit}.csv').set_index('id').loc[tr.id]
            v=pd.read_csv(A/f'features/held_{fit}.csv').set_index('id');t=pd.read_csv(A/f'features/test_{fit}.csv').set_index('id').loc[te.id]
            cols=list(x.columns)
            if kind=='relations':cols=[c for c in cols if c not in PIXEL]
            if kind=='pixels':cols=[c for c in cols if c in PIXEL or c in ['log_width','log_height','aspect','size_width_z','size_height_z']]
            pp=np.zeros(len(te))
            for cls in CL:
                m=tr.class_name.eq(cls).to_numpy();y=tr.loc[m,'error'].to_numpy(int);weights=tr.loc[m,'weight'].to_numpy(float)
                xx=x.loc[m,cols];groups=tr.loc[m,'frame_id'].to_numpy()
                def make():
                    if algorithm=='logreg':return make_pipeline(StandardScaler(),LogisticRegression(C=.08,max_iter=1500,class_weight='balanced'))
                    return CatBoostClassifier(iterations=260,depth=3,learning_rate=.035,l2_leaf_reg=20,random_seed=2301,
                        thread_count=2,verbose=False,allow_writing_files=False)
                def train(model,xx,yy,ww):
                    if algorithm=='logreg':model.fit(xx,yy,logisticregression__sample_weight=ww)
                    else:model.fit(xx,yy,sample_weight=ww)
                raw=np.zeros(len(y))
                for ii,jj in GroupKFold(3).split(xx,y,groups):
                    model=make();train(model,xx.iloc[ii],y[ii],weights[ii]);raw[jj]=model.predict_proba(xx.iloc[jj])[:,1]
                logits=lambda p:np.log(np.clip(p,1e-5,1-1e-5)/np.clip(1-p,1e-5,1))[:,None]
                cal=LogisticRegression(C=.5,max_iter=1000);cal.fit(logits(raw),y,sample_weight=weights)
                model=make();train(model,xx,y,weights)
                vm=q.day.eq(held)&q.class_name.eq(cls);tm=te.class_name.eq(cls)
                pv=cal.predict_proba(logits(model.predict_proba(v.loc[q.loc[vm,'id'],cols])[:,1]))[:,1]
                pt=cal.predict_proba(logits(model.predict_proba(t.loc[te.loc[tm,'id'],cols])[:,1]))[:,1]
                po[vm]=pv;pp[tm]=pt
                standalone.append(dict(variant=kind+'_'+algorithm,fit=fit,held=held,class_name=cls,ntrain=int(m.sum()),train_frames=len(set(groups)),ap=float(AP(q.loc[vm,'error'],pv))))
                joblib.dump(dict(model=model,calibrator=cal,columns=cols,fit_day=fit,training_frames=sorted(set(groups))),A/f'models/{kind}_{algorithm}_{cls}_{fit}.joblib',compress=3)
                print('MODEL',kind,algorithm,fit,cls,standalone[-1]['ap'],flush=True)
            testprob.append(pp)
        pt=np.mean(testprob,axis=0)
        pd.DataFrame(standalone).to_csv(A/f'results/standalone_{kind}_{algorithm}.csv',index=False)
        for split,rows,p in [('oof',q,po),('test',te,pt)]:
            pd.DataFrame(dict(id=rows.id,probability=p)).to_csv(A/f'features/model_{kind}_{algorithm}_{split}.csv',index=False)
        for cls in CL:
            for direction in ['both','rescue','penalty']:
                for weight in [.2,.4]:
                    scores=[]
                    for split,rows,base,p in [('oof',q,base_oof,po),('test',te,current,pt)]:
                        f=fs[split];gate=rows.class_name.eq(cls).to_numpy()
                        lower=(p<base)&(p<.4);higher=(p>base)&(p>.7)
                        vv,zz=veto(split,rows)
                        if cls=='nude_hands':lower &= ~vv & (zz.semantic_hand.to_numpy()>=.8) & (f.wrist_distance.to_numpy()<1.5)
                        if cls=='person':
                            lower &= (f.body_outside_fraction.to_numpy()<.05)&(f.head_count.to_numpy()==1)
                            higher &= (f.head_count.to_numpy()>1)|((f.head_count.to_numpy()==1)&((f.body_point_iou_q75.to_numpy()<.65)|(f.seat_iou_max.to_numpy()<.65)))
                        if cls in ['cup','tray']:
                            sem=align(ROOT/f'rascar_v16/semantic_both_{split}.csv',rows,'p_'+cls)
                            st=align(ROOT/f'rascar_v21/object_structure_{split}.csv',rows)
                            lower &= (sem>=.8)&st.distinct_instances.le(1).to_numpy()&~st.ambiguous_duplicate.astype(bool).to_numpy()
                        if split=='test':
                            lower &= np.maximum(*testprob)<.5;higher &= np.minimum(*testprob)>.6
                        gate &= lower if direction=='rescue' else higher if direction=='penalty' else lower|higher
                        score=base.copy();score[gate]+=weight*(p[gate]-base[gate]);scores.append(score)
                    register(f'{kind}_{algorithm}_{cls}_{direction}_w{weight}',*scores,
                        dict(type='cross_day_CV_context_model',features=kind,algorithm=algorithm,class_name=cls,direction=direction,weight=weight))

if __name__=='__main__':
    register('baseline_v22',base_oof,current,dict(lb=.95610,primary_oof='without inherited single-query correction'))
    rules();models()
    rows=[json.loads(p.read_text()) for p in (A/'results').glob('*.json')]
    pd.DataFrame(rows).sort_values('pooled_ap',ascending=False).to_csv(A/'results/experiment_registry.csv',index=False)
