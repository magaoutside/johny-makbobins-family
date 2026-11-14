"""Reproduce using the existing project data / model caches; no network or LB.

Default: regenerate score diagnostics/package. --refit: reproduce candidate.
--all: additionally repeat every research family with frozen pool reviews.
"""
from pathlib import Path
import argparse, subprocess, sys, os

def main():
    p=argparse.ArgumentParser();p.add_argument('--refit',action='store_true');p.add_argument('--all',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent;stages=[]
    if a.all:
        for path in ['inputs/blind_review.json','inputs/near_hand_review.json']:
            assert (root/path).exists(), 'Reviewed training labels required: '+path
        stages += [('appearance.py',),('correspondence.py',),('evaluate_correspondence.py',),('sift_transfer.py',),('evaluate_correspondence.py','--sift')]
    if a.refit or a.all:stages += [('refit.py',),('refit_guard.py',)]
    if a.all:stages += [('object_refit_guard.py',),('hand_hardneg.py',),('near_hand_negatives.py',),('fit_near_hand.py',),('negative_occupancy.py',)]
    stages += [('stability.py',),('finalize.py',)]
    env=dict(os.environ);env['HF_HUB_OFFLINE']='1';env['MPLCONFIGDIR']=str(root/'features/mplconfig')
    for script,*args in stages:
        print('RUN',script,*args,flush=True)
        subprocess.run([sys.executable,str(root/script),*args],check=True,cwd=root.parent,env=env)

if __name__=='__main__':main()
