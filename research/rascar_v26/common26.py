"""V26 experiments; the confirmed V22 and all V25 artifacts stay immutable."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rascar_v25'))
import common25 as c25
from common25 import *
A = ROOT / 'rascar_v26'
for folder in ['inputs','models','features','results/oof','results/submissions','results/reviews','audit','logs']:
    (A / folder).mkdir(parents=True, exist_ok=True)
c25.A = A

