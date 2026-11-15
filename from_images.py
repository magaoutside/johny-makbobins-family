"""Full selected pipeline from original photos, CSVs and published training annotations."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from rascar.core import ROOT
from rascar.upstream_raw import metadata,install_models,foundation_bridge,worker
from rebuild_upstream import initialize,run_stage,export


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=ROOT/'cache/competition')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/from_images')
    p.add_argument('--weights',type=Path,default=ROOT/'cache/weights')
    p.add_argument('--models',type=Path,default=ROOT/'models')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--python-cv',type=Path,default=Path(sys.executable))
    p.add_argument('--python-vlm',type=Path,default=ROOT/'.venv-vlm/bin/python')
    p.add_argument('--python-sam3',type=Path,default=ROOT/'.venv-sam3/bin/python')
    p.add_argument('--python-sam2',type=Path,default=ROOT/'.venv-sam2/bin/python')
    p.add_argument('--retrain',action='store_true',help='Train all selected models into a new output folder')
    p.add_argument('--start',choices=['metadata','foundation','local','guards','tta','body','sahi','kernel','boundary','final'],default='metadata')
    a=p.parse_args()
    if a.retrain and a.start=='metadata' and a.out.exists():p.error('--retrain requires a new --out directory; keep previous training runs intact')
    for name in ['raw','out','weights','models','python_cv','python_vlm','python_sam3','python_sam2']:setattr(a,name,getattr(a,name).resolve())
    work=initialize(a.out/'upstream',cached=False);early_models=a.out/'early_models' if a.retrain else ROOT/'upstream/weights';data=a.out/'data';det=data/'detections';cache=a.out/'features';foundation=a.out/'foundation'
    for d in [data,det,cache,foundation]:d.mkdir(parents=True,exist_ok=True)
    order=['metadata','foundation','local','guards','tta','body','sahi','kernel','boundary','final'];start=order.index(a.start)
    def active(stage):return order.index(stage)>=start
    def prep(stage,python,out=det):
        subprocess.run([str(python),str(ROOT/'prepare.py'),stage,'--raw',str(a.raw),'--data',str(data),'--out',str(out),'--weights',str(a.weights),'--detections',str(det),'--device',a.device],check=True)
    def job(stage,script,args=(),python=None):worker(work,stage,script,args,python or a.python_cv,a.raw,a.weights,a.device)
    if active('metadata'):
        metadata(a.raw,data,work)
        if not a.retrain:install_models(work,early_models)
        # Foundation checkpoint paths used by the historical detector workers.
        for name in ['yolo26s.pt']:
            dest=work/'local_evidence/weights'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.weights/'detector'/name,dest)
    if active('foundation'):
        prep('sam3',a.python_sam3)
        prep('wevis-full',a.python_vlm);prep('wevis-zoom',a.python_vlm)
        prep('retrieval',a.python_vlm,cache)
        from rascar.retrieval import assemble
        assemble(data,cache,det,foundation)
        from rascar.foundation_models import detector,geometry
        # This script is launched in the CV environment.
        detector(data,a.raw,a.weights,early_models,det,a.device,train=a.retrain)
        geometry(data,foundation,det,early_models,foundation,train=a.retrain)
        for fit in ['2026-10-25','2026-10-26','all']:
            shutil.copy2(early_models/f'generic_{fit}.pt',work/'local_evidence/weights'/f'{fit}.pt')
        foundation_bridge(data,work,foundation)
        for name in ['reference_scores.csv','fused_references.csv']:shutil.copy2(foundation/name,det/name)
    if active('local'):
        job('local_evidence','extract_local.py');job('local_evidence','extract_pose.py')
        job('local_evidence','extract_semantics.py',python=a.python_vlm)
        for fit in ['2026-10-25','2026-10-26','all']:job('local_evidence','train_hand_specialist.py',[fit,a.device.removeprefix('cuda:')])
        run_stage(work,'local_evidence')
        if a.retrain:
            for fit in ['2026-10-25','2026-10-26','all']:
                source=work/'local_evidence/hand_specialist'/fit/'train/weights/last.pt'
                targets=[work/'seat_and_tta/weights'/f'hand{fit[-2:] if fit!="all" else "all"}_last.pt']
                if fit!='all':targets.append(work/'modes_and_sahi/weights'/f'hand_{fit}.pt')
                for dest in targets:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    if active('guards'):
        prep('semantic',a.python_vlm,work/'geometry_guards')
        prep('wevis-heads',a.python_vlm);prep('wevis-hands',a.python_vlm);prep('pose',a.python_cv)
        for source,target in [('heads.csv','wevis_heads_references.csv'),('hands.csv','wevis_hands_references.csv'),('poses.csv','multiview_pose.csv')]:shutil.copy2(det/source,work/'geometry_guards'/target)
        run_stage(work,'geometry_guards')
    if active('tta'):
        for fit in ['2026-10-25','2026-10-26','all']:job('seat_and_tta','gpu_worker.py',[fit,a.device.removeprefix('cuda:')])
        run_stage(work,'seat_and_tta')
    if active('body'):run_stage(work,'body_and_hands')
    if active('sahi'):
        for fit in ['2026-10-25','2026-10-26']:job('modes_and_sahi','gpu_yolo_hands.py',[fit,a.device.removeprefix('cuda:')])
        run_stage(work,'modes_and_sahi')
    if active('kernel'):run_stage(work,'kernel_quality')
    if active('boundary'):
        job('boundary_review','build_spatial_records.py');job('boundary_review','gpu_spatial.py')
        for fit in ['2026-10-25','2026-10-26']:job('boundary_review','gpu_hand_train.py',[fit])
        run_stage(work,'boundary_review')
    if active('final'):
        export(work,ROOT/'data',data)
        prep('geometry',a.python_cv,data) # passes explicit detection directory below
        prep('roi',a.python_cv,data);prep('dense-small',a.python_cv,cache)
        if a.retrain:a.models=a.out/'trained_models'
        subprocess.run([str(a.python_cv),str(ROOT/'rebuild_baseline.py'),'--data',str(data),'--cache',str(cache),'--models',str(a.models),'--out',str(data),'--device',a.device]+(['--train'] if a.retrain else []),check=True)
        final=(ROOT/'rascar/blend.py').exists()
        if final:prep('sam2',a.python_sam2,data);prep('dense',a.python_cv,cache)
        if a.retrain:
            subprocess.run([str(a.python_cv),str(ROOT/'train.py'),'--component','all' if final else 'prototype','--data',str(data),'--cache',str(cache),'--weights',str(a.weights),'--out',str(a.models),'--device',a.device],check=True)
        cmd=[str(a.python_cv),str(ROOT/'predict.py'),'--data',str(data),'--models',str(a.models),'--out',str(a.out/'predictions')]
        if final:cmd+=['--mode','models','--cache',str(cache),'--weights',str(a.weights),'--device',a.device]
        subprocess.run(cmd,check=True)
    print(f'Final artifacts: {a.out}',flush=True)

if __name__=='__main__':main()
