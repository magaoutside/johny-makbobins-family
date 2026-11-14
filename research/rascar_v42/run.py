from pathlib import Path
import subprocess,sys
A=Path(__file__).resolve().parent
jobs=[('extract',['extract.py'])]+[(v,['train.py','--variant',v]) for v in ['tight','dual','dual_pool']]
for name,args in jobs:
 print('START',name,flush=True)
 with (A/'logs'/f'{name}.log').open('w') as log:subprocess.run([sys.executable,str(A/args[0])]+args[1:],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('DONE',name,flush=True)
print('ALL CLASSIFIERS DONE',flush=True)
