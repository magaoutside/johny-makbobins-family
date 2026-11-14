from pathlib import Path
import time,subprocess,sys
A=Path(__file__).resolve().parent
deadline=time.time()+1800
while not (A/'features/vit_l_predictions_s4101.csv').exists():
 if time.time()>deadline:raise RuntimeError('Vision run did not finish in 30 minutes; inspect logs.')
 time.sleep(5)
for key in ['clip','vit_b','vit_l']:
 print('GEOMETRY',key,flush=True)
 with (A/f'logs/{key}_geometry.log').open('a') as f:subprocess.run([sys.executable,'-u',str(A/'train_geometry.py'),key],stdout=f,stderr=subprocess.STDOUT,check=True)
print('YOLO',flush=True)
with (A/'logs/yolo.log').open('a') as f:subprocess.run([str(A/'env/bin/python'),'-u',str(A/'yolo_experiment.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('EVALUATE',flush=True)
with (A/'logs/evaluate.log').open('a') as f:subprocess.run([sys.executable,'-u',str(A/'evaluate.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('FOLLOWUP COMPLETE',flush=True)
