"""Validate release shape, frozen baseline, split coverage and portable replay."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
import numpy as np
import pandas as pd

A=Path(__file__).resolve().parent
expected_base='f9b07bfcb4e92d85bcac64635c0303e2e650a6310cfe47b746d1b9c149c0babe'
assert hashlib.sha256((A.parent/'rascar_v20/submission_v20_main.csv').read_bytes()).hexdigest()==expected_base
assert hashlib.sha256((A/'baseline_lb095417.csv').read_bytes()).hexdigest()==expected_base
baseline=pd.read_csv(A/'baseline_lb095417.csv')
q=pd.read_csv(A/'inputs/baseline_oof.csv')
ledger=pd.read_csv(A/'kernel_training_ledger.csv')
assert len(ledger)==24 and (ledger.fit!=ledger.held).all()
assert set(ledger.fit)=={'2026-10-25','2026-10-26'}
releases={}
for name in ['main','conservative','seat_experiment']:
    p=A/f'submission_v21_{name}.csv';f=pd.read_csv(p)
    assert list(f.columns)==['id','score'] and f.id.tolist()==baseline.id.tolist()
    assert len(f)==362 and f.id.is_unique and np.isfinite(f.score).all() and f.score.between(0,1).all()
    expected=json.loads((A/f'results/release_{name}.json').read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256']
    releases[name]=dict(rows=len(f),changed=int(np.sum(abs(f.score-baseline.score)>1e-9)),sha256=expected['sha256'])

with tempfile.TemporaryDirectory(prefix='rascar-v21-replay-') as td:
    isolated=Path(td)
    shutil.copy2(A/'replay.py',isolated/'replay.py')
    shutil.copytree(A/'replay_inputs',isolated/'replay_inputs')
    proc=subprocess.run([sys.executable,str(isolated/'replay.py')],cwd=isolated,text=True,capture_output=True,check=True)
    verification=json.loads((isolated/'replayed/verification.json').read_text())

class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        for key in ['href','src']:
            if key in a and not a[key].startswith(('http:','https:','#')):self.paths.append(a[key])
links=Links();links.feed((A/'report.html').read_text())
for p in links.paths:assert (A/p).exists(),p
review=pd.read_csv(A/'audit/visual_review_decisions.csv')
assert len(review)==69 and review.id.is_unique
new=pd.read_csv(A/'results/oof/release_main.csv').set_index('id').loc[q.id,'score'].to_numpy()
changed=q[abs(new-q.score.to_numpy())>1e-9]
assert len(changed)==11 and len(changed[changed.class_name.eq('nude_hands')])==2
assert changed.loc[changed.class_name.eq('nude_hands'),'error'].eq(1).all()
record=dict(baseline_unchanged=True,baseline_sha256=expected_base,releases=releases,
    new_quality_models=24,outer_fit_held_day_disjoint=True,
    isolated_replay=verification,local_report_links_checked=len(links.paths),visual_cases_reviewed=len(review),
    changed_oof_by_class=changed.class_name.value_counts().to_dict(),
    note='Checks validate artifacts and new-model split bookkeeping; not an independent validation dataset or LB guarantee.')
(A/'verification.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
