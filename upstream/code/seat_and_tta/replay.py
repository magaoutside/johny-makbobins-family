"""Rebuild released CSVs from saved CV predictions and fitted seat features.

No GPU training is performed here. The training code, initial checkpoint
payload, refined checkpoints and logs are retained separately in this bundle.
"""
from common17 import *
import subprocess

for script in ['seat_guards.py','combine_seats.py','evaluate_hand_tta.py','tta_contradiction.py','guard_tta.py','hybrid_tta.py','merged_hand_ablation.py','build_release.py']:
    subprocess.run([sys.executable,str(A/script)],cwd=A,check=True)
print('Selected numerical stages complete',flush=True)
