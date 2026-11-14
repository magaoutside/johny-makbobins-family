from pathlib import Path
import time,subprocess,sys
A=Path(__file__).resolve().parent;deadline=time.time()+1200
while not (A/'pool_expansion/evidence.csv').exists():
 if time.time()>deadline:raise RuntimeError('Pool mining not complete; inspect logs.')
 time.sleep(5)
for key in ['vit_b','vit_l']:
 for flags in [[],['--transductive']]:
  variant='transductive' if flags else 'expanded';print('RUN',key,variant,flush=True)
  with (A/f'logs/{key}_{variant}.log').open('a') as f:subprocess.run([sys.executable,'-u',str(A/'train_expanded.py'),key,*flags],stdout=f,stderr=subprocess.STDOUT,check=True)
print('EXPANSION COMPLETE',flush=True)
