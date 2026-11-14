
from pathlib import Path
import shutil,kagglehub
_competition_roots=[p.parent for p in Path('/kaggle/input').rglob('machine_frames.csv') if (p.parent/'machine_predictions.csv').exists()]
assert len(_competition_roots)==1, ('Attach Rascar competition Input',_competition_roots)
_competition_root=_competition_roots[0]
def _mounted_competition_download(handle,path=None,output_dir=None,**kwargs):
 assert handle=='rascar-hack-case-1'
 if path is None:return str(_competition_root)
 src=_competition_root/path
 if not src.is_file():
  matches=list(_competition_root.rglob(Path(path).name));assert len(matches)==1,(path,len(matches));src=matches[0]
 if output_dir:
  dst=Path(output_dir)/path;dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():shutil.copy2(src,dst)
  return str(dst)
 return str(src)
kagglehub.competition_download=_mounted_competition_download
print('MOUNTED_COMPETITION',str(_competition_root),flush=True)
