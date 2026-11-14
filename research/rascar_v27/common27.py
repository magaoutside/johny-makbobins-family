from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'rascar_v25'))
import common25 as c25
from common25 import *
A=ROOT/'rascar_v27'
for directory in ['inputs','models','features','results/oof','results/submissions','results/reviews','audit','logs']:
    (A/directory).mkdir(parents=True,exist_ok=True)
c25.A=A
