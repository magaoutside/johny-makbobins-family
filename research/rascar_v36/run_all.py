"""Reproduce V36 using existing project data/caches and downloaded public weights.
Does not download data, annotate images, or submit to Kaggle.
V34/V36 human-presence review JSON files are fixed input artifacts.
"""
from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().parent
steps=[['dense_votes.py','--kinds','dense','center','center_focal'],['evaluate_votes.py'],['evaluate_focal.py'],['identity_check.py'],['multiscale.py'],['evaluate_multiscale.py'],['compose.py'],['replicate.py'],['seed_consensus.py'],['strict_seeds.py'],['crowded_hands.py'],['verify_geometry.py'],['siglip_semantic.py'],['evaluate_blind_pool.py'],['augment_pool_semantics.py'],['blend.py'],['pose_guard.py'],['stability.py'],['release_audit.py'],['visualize_semantic.py'],['visualize_pose_guard.py'],['finalize.py']]
for step in steps:
 print('RUN',*step,flush=True);subprocess.run([sys.executable,str(p/step[0]),*step[1:]],cwd=p.parent,check=True)
