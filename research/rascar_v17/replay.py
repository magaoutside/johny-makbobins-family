"""Rebuild released CSVs from saved CV predictions and fitted seat features.

No GPU training is performed here. The training code, initial checkpoint
payload, refined checkpoints and logs are retained separately in this bundle.
"""
from common17 import *
import subprocess
manifest=json.loads((A/'release_manifest.json').read_text())
assert hashlib.sha256((V16/'baseline_submission_0.92822.csv').read_bytes()).hexdigest()==manifest['baseline_sha256']
assert hashlib.sha256((V16/'submission_v16_targeted_short.csv').read_bytes()).hexdigest()=='299df301ba6ac84791165198e64896b3c8fa504403a3ae79177a42bf575bb347'
for script in ['seat_guards.py','combine_seats.py','evaluate_hand_tta.py','tta_contradiction.py','guard_tta.py','hybrid_tta.py','merged_hand_ablation.py','build_release.py']:
    subprocess.run([sys.executable,str(A/script)],cwd=A,check=True)
for name,expected in manifest['release_experiments'].items():
    path=R/'submissions'/f'submission_{name}.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256'],name
    actual=json.loads((R/f'{name}.json').read_text())
    for field in ['pooled_ap','day_25','day_26','ap_person','ap_nude_hands']:
        assert abs(actual[field]-expected[field])<1e-12,(name,field)
print('REPLAY_OK: exact submission hashes and opposite-day validation metrics reproduced.')
