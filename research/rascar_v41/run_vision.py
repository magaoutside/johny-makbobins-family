from pathlib import Path
import subprocess,sys,time
A=Path(__file__).resolve().parent
for key in ['clip','vit_b','vit_l']:
 for script in ['extract.py','train.py']:
  if script=='extract.py' and (A/f'features/{key}_extraction.json').exists():continue
  print('RUN',key,script,flush=True)
  with (A/f'logs/{key}_{script[:-3]}.log').open('a') as f:
   subprocess.run([sys.executable,'-u',str(A/script),key],stdout=f,stderr=subprocess.STDOUT,check=True)
print('ALL VISION EXPERIMENTS COMPLETE',flush=True)
