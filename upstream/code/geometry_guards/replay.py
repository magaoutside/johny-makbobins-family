"""CPU replay from saved CV inference and fitted-head predictions.

This does not retrain the V15 baseline or rerun the GPU foundation models.
Those source scripts and the complete V16 feature/model caches are retained.
"""
from common16 import *
import subprocess

def run(script):
    subprocess.run([sys.executable,str(V16/script)],cwd=V16,check=True)



# Recreate semantic and structural components from their saved fitted/frozen
# predictions, instead of trusting previously assembled component submissions.
for view in ['tight','context','both']:
    a=pd.read_csv(V16/f'semantic_{view}_oof.csv').set_index('id').loc[q.id]
    b=pd.read_csv(V16/f'semantic_{view}_test.csv').set_index('id').loc[te.id]
    x=np.clip((a.p_background-a.p_person-.1)/.7,0,1).to_numpy()*q.class_name.eq('person').to_numpy()
    y=np.clip((b.p_background-b.p_person-.1)/.7,0,1).to_numpy()*te.class_name.eq('person').to_numpy()
    for w in [.25,.5]:register(f'semantic_{view}_person_{w}',q.score+w*(1-q.score)*x,current+w*(1-current)*y,dict(view=view,weight=w))
for signal in ['head_near_outside','two_heads']:
    sig=[]
    for rows,test in [(q,False),(te,True)]:
        f=pd.read_csv(V16/f'wevis_features_{"test" if test else "oof"}.csv').set_index('id').loc[rows.id]
        s=f.head_near_outside.to_numpy() if signal=='head_near_outside' else (f.head_count.to_numpy()>=2)
        sig.append(s*rows.class_name.eq('person').to_numpy())
    for w in [.25,.5]:register(f'wevis_{signal}_{w}',q.score+w*(1-q.score)*sig[0],current+w*(1-current)*sig[1],dict(signal=signal,weight=w))
for script in ['guarded_dino.py','calibrate_hand_boxes.py','head_anchored_geometry.py','combine.py','guarded_structure.py']:
    run(script)
print('Selected numerical stages complete',flush=True)
