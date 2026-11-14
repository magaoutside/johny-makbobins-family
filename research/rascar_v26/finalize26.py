from common26 import *
import shutil, itertools, zipfile, html
import render23
from perspective_tray import CameraMetric, CALIBRATION, transform

def main(mode='physical'):
    hands=align(A/'submission_v26_reviewed_hands.csv',te,'score')
    ho=align(A/'results/oof/accepted_reviewed_hands.csv',q,'score')
    trayname='tray_query_oriented_monotone_grouped_confirm_e0.5_iou0.8_w0.2' if mode=='monotone' else 'tray_query_oriented_confirm_e0.5_iou0.7_w0.2'
    combined_name='hands_and_monotone_trays_research' if mode=='monotone' else 'hands_and_perspective_trays_research'
    feature_prefix='tray_query_oriented_monotone_grouped' if mode=='monotone' else 'tray_query_oriented'
    trays=align(A/f'results/submissions/submission_{trayname}.csv',te,'score')
    to=align(A/f'results/oof/{trayname}.csv',q,'score')
    combined=np.clip(hands+trays-current,0,1);co=np.clip(ho+to-base_oof,0,1)
    result=register(combined_name,co,combined,dict(type='reviewed_hands_plus_directional_rectangle_confirmation',hand_source='V25 retained after eight-row review',tray_source=trayname,person_probe_excluded=True,validation_note='No demonstrated tray AP gain; selected during development including visual test audit, not an independent final evaluation'))
    target=A/f'submission_v26_{combined_name}.csv'
    shutil.copyfile(A/f'results/submissions/submission_{combined_name}.csv',target)
    render23.A=A
    for name in [trayname,combined_name]:
        d=pd.read_csv(A/f'results/reviews/{name}.csv')
        refs=(align(A/f'features/{feature_prefix}_test_{DAYS[0]}.csv',te)+align(A/f'features/{feature_prefix}_test_{DAYS[1]}.csv',te))/2
        render23.render(d[(d.delta.abs()>1e-9)&d.class_name.eq('tray')],name,ref=refs)
    d=q.assign(baseline_v22=base_oof,new_score=to)
    ref=pd.concat([pd.read_csv(A/f'features/{feature_prefix}_held_{fit}.csv') for fit in DAYS]).set_index('id').loc[q.id]
    render23.render(d[abs(to-base_oof)>1e-9],'tray_changed_oof_'+mode,gt=True,ref=ref)
    # Explicitly retain disagreement with the old nonofficial visual audit.
    r=audit[audit.review_no.eq(6)].iloc[0];manual=np.array([724.,240.,780.,301.]);query=r[B].to_numpy(float)
    ranges={}
    for delta in [1,2,3]:
        values=[float(iou(query,manual+v)[0,0]) for v in itertools.product([-delta,delta],repeat=4)]
        ranges[str(delta)]=[min(values),max(values)]
    note=dict(review_no=6,old_visual_label='likely_error / tray_undercoverage',new_review='uncertain under official IoU rule; do not treat visible undercoverage as automatically erroneous',
        approximate_visual_bbox=manual.tolist(),approximate_iou=float(iou(query,manual)[0,0]),coordinate_perturbation_ranges=ranges,
        not_ground_truth=True,not_used_for_training_or_threshold_fit=True,official_correct_iou=.6,
        comment='The original query omits a small lower strip but includes the tray broadly. ±3 pixel uncertainty crosses 0.6, so neither success nor failure is proved. Frozen audit remains unchanged.')
    (A/'audit/tray6_geometry_reassessment.json').write_text(json.dumps(note,indent=2))
    benchmark=[]
    for p in sorted((A/'results').glob('tray_completion_benchmark_*.csv')):
        d=pd.read_csv(p)
        for fraction,g in d.groupby('fraction'):
            benchmark.append(dict(method=p.stem.removeprefix('tray_completion_benchmark_'),fraction=fraction,n=len(g),before=g.before.mean(),after=g.after.mean(),changed=int(g.changed.sum()),improved=int((g.after>g.before+1e-8).sum()),degraded=int((g.after<g.before-1e-8).sum())))
    pd.DataFrame(benchmark).to_csv(A/'results/completion_summary.csv',index=False)
    registry=[]
    for p in sorted((A/'results').glob('*.json')):
        d=json.loads(p.read_text())
        if 'experiment' in d:registry.append({k:v for k,v in d.items() if not isinstance(v,(dict,list))})
    pd.DataFrame(registry).to_csv(A/'results/experiment_registry.csv',index=False)
    checked=0
    for p in (A/'results/submissions').glob('*.csv'):
        s=pd.read_csv(p)
        assert s.id.tolist()==te.id.tolist() and s.id.is_unique
        assert np.isfinite(s.score).all() and s.score.between(0,1).all()
        checked+=1
    for cls in ['person','cup']:
        mask=te.class_name.eq(cls)
        assert np.array_equal(combined[mask],current[mask])
    assert np.max(abs((co-base_oof)[~q.class_name.isin(['tray','nude_hands'])]))==0
    maxerr=0.
    for fit in DAYS:
        cam=CameraMetric(fit,'physical')
        for H,Hi in cam.maps:
            points=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.3,.7]])
            err=np.max(abs(transform(transform(points,H),Hi)-points));maxerr=max(maxerr,float(err))
    assert maxerr<1e-10
    for fit in DAYS:
        cam=CameraMetric(fit,'monotone')
        for xx in [200,400,600,800,1000,1200]:
            sizes=[]
            for yy in [210,300,390]:
                p=cam.rectangle([xx,yy],.5,.3,np.deg2rad([0,30,60,90]))
                sizes.append(p.max(1)-p.min(1))
            assert (np.diff(sizes,axis=0)>0).all()
    assert hashlib.sha256((V22/'submission_v22_candidate.csv').read_bytes()).hexdigest()=='44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c'
    summary=dict(baseline_lb=.95610,reported_person_probe_lb=.95610,baseline_oof=metric(q,base_oof),retained_hands=metric(q,ho),research_combined=result,
        retained_hand_rows=int((abs(hands-current)>1e-9).sum()),research_tray_rows=int((abs(trays-current)>1e-9).sum()),total_experiments=checked,
        hand_decision='Retain in working candidate, not a new confirmed LB best',tray_decision='Save research candidate; image-level support but no demonstrated AP gain',submitted=False)
    (A/'summary.json').write_text(json.dumps(summary,indent=2))
    (A/'checks.json').write_text(json.dumps(dict(submissions_checked=checked,baseline_unchanged=True,person_and_cup_unchanged=True,homography_roundtrip_max_error=maxerr,monotone_width_height_at_six_positions_four_angles=True,combination_oof_reproduced=True),indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main(*(sys.argv[1:] or []))
