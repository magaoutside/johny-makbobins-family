import subprocess,sys
from pathlib import Path
O=Path("/kaggle/working/rascar_v58")
for bb in ["vit_s","vit_b"]:
 for seed in [3101,3102,3103]:
  result=subprocess.run([sys.executable,"-u",str(O/"gpu_finetune.py"),sys.argv[1],sys.argv[2],str(seed),bb])
  assert result.returncode==0
print("ALL_FINE_DONE",flush=True)
