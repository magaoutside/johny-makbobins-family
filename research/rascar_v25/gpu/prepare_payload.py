import sys,zipfile,base64
from pathlib import Path
A=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(A));from common25 import *
from features23 import refs
manual=h.copy();manual['id']=['manual_'+str(i) for i in manual.index];manual['split']='manual'
pool=pd.read_csv(V22/'inputs/reviewed_pool_hands.csv').assign(split='reviewed_pool')
# All 170 manual/test frames plus 13 reviewed pool boxes. Ground-truth boxes used
# only to define local benchmark crops, never person proposals or inference tuning.
qq=pd.concat([q.assign(split='oof'),te.assign(split='test'),manual[manual.class_name.eq('nude_hands')],pool],ignore_index=True)
qq.to_csv(A/'gpu/queries.csv',index=False)
refs[refs.class_name.eq('person')&refs.confidence.ge(.6)].to_csv(A/'gpu/persons.csv',index=False)
with zipfile.ZipFile(A/'gpu/benchmark_payload.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in ['queries.csv','persons.csv','benchmark.py']:z.write(A/'gpu'/n,n)
b64=base64.b64encode((A/'gpu/benchmark_payload.zip').read_bytes()).decode()
cell="from pathlib import Path\nimport base64,io,zipfile,subprocess,sys,os\nA=Path('/kaggle/working/rascar_v25')\nzipfile.ZipFile(io.BytesIO(base64.b64decode("+repr(b64)+"))).extractall(A)\npose_jobs={}\nfor gpu,kind in enumerate(['rtmw','wilor_detector']):\n pose_jobs[kind]=subprocess.Popen([sys.executable,'-u',str(A/'benchmark.py'),kind],stdout=open(A/(kind+'.log'),'w'),stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)))\nprint('POSE_JOBS_STARTED',{k:v.pid for k,v in pose_jobs.items()})\n"
(A/'gpu/launch_cell.py').write_text(cell)
print('PAYLOAD',len(cell),'queries',len(qq),'frames',qq.frame_id.nunique())
