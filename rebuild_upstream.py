"""Rebuild the inherited score from selected early models and detector evidence."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from rascar.core import ROOT,queries

BRIDGES={
 'local_evidence':[('local_evidence/results/submissions/submission_final_hands0_pose0.2.csv','geometry_guards/baseline_submission_0.92822.csv')],
 'geometry_guards':[('geometry_guards/results/submissions/submission_final_targeted_short_with_contradiction.csv','geometry_guards/submission_v16_targeted_short.csv')],
 'seat_and_tta':[('seat_and_tta/results/submissions/submission_release_balanced.csv','body_and_hands/baseline_lb095180.csv')],
 'body_and_hands':[('body_and_hands/results/oof/release_main.csv','modes_and_sahi/inputs/baseline_oof.csv'),('body_and_hands/results/submissions/submission_release_main.csv','modes_and_sahi/baseline_lb095259.csv')],
 'modes_and_sahi':[(f'modes_and_sahi/results/{src}',f'{stage}/{dst}') for stage in ['kernel_quality','boundary_review'] for src,dst in [('oof/release_main.csv','inputs/baseline_oof.csv'),('submissions/submission_release_main.csv','baseline_lb095417.csv')]]}


def initialize(work,cached=True):
    work=Path(work).resolve()
    if work==ROOT or work.is_relative_to(ROOT/'upstream'):
        raise ValueError('Use a separate output workspace, not the bundled source/artifact folders')
    work.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'upstream/code',work,dirs_exist_ok=True)
    if cached:shutil.copytree(ROOT/'upstream/data',work,dirs_exist_ok=True)
    return work


def run_stage(work,stage,python=sys.executable):
    plan=json.loads((ROOT/'upstream/plan.json').read_text())['stages']
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    (work/stage/'models').mkdir(parents=True,exist_ok=True)
    logs=work/'logs';logs.mkdir(exist_ok=True)
    for script in plan[stage]:
        path=logs/f'{stage}_{script}.log'
        with path.open('w') as log:
            r=subprocess.run([python,str(work/stage/script)],cwd=work/stage,env=env,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:
            print(path.read_text()[-5000:]);raise RuntimeError(f'{stage}/{script} failed; log: {path}')
        print(f'OK {stage}/{script}',flush=True)
    for src,dst in BRIDGES.get(stage,[]):
        (work/dst).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(work/src,work/dst)


def export(work,data,out):
    rows=queries(data);pieces=[]
    def aligned(path,ids):return pd.read_csv(path).set_index('id').loc[ids,'score'].to_numpy()
    for split in ['oof','test']:
        r=rows[rows.split.eq(split)]
        part='oof/final_reviewed.csv' if split=='oof' else 'submissions/submission_final_reviewed.csv'
        score=aligned(work/'boundary_review/results'/part,r.id)
        if split=='oof':
            inherited=aligned(work/'body_and_hands/results/oof/release_main.csv',r.id)-aligned(work/'body_and_hands/results/oof/release_main_no_inherited_contradiction.csv',r.id)
            score=score-inherited
        pieces.append(pd.DataFrame(dict(id=r.id,score=score)))
    result=pd.concat(pieces,ignore_index=True)
    if not np.isfinite(result.score).all() or not result.score.between(0,1).all():raise ValueError('Invalid rebuilt score')
    out.mkdir(parents=True,exist_ok=True);result.to_csv(out/'guarded_scores.csv',index=False)
    original=pd.read_csv(Path(data)/'guarded_scores.csv').set_index('id').loc[result.id,'score'].to_numpy()
    drift=float(np.max(np.abs(result.score.to_numpy()-original)))
    (out/'upstream_check.json').write_text(json.dumps(dict(rows=len(result),max_abs_difference=drift),indent=2)+'\n')
    print(f'Saved {out/"guarded_scores.csv"}; maximum difference from bundled score: {drift:.3g}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=ROOT/'outputs/upstream')
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/rebuilt_features')
    p.add_argument('--stages',nargs='+',choices=list(BRIDGES)+['kernel_quality','boundary_review'])
    p.add_argument('--existing',action='store_true',help='Use a workspace populated by raw GPU stages; do not copy bundled feature caches')
    a=p.parse_args();work=initialize(a.work,cached=not a.existing)
    plan=json.loads((ROOT/'upstream/plan.json').read_text())['stages']
    for stage in a.stages or plan:run_stage(work,stage)
    if not a.stages or a.stages[-1]=='boundary_review':export(work,a.data,a.out)

if __name__=='__main__':main()
