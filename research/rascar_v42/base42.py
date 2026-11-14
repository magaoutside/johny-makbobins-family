from pathlib import Path
import os,sys,json,hashlib
A=Path(__file__).resolve().parent;ROOT=A.parent
os.environ['MPLCONFIGDIR']=str(A/'model_cache/mpl')
sys.path.insert(0,str(ROOT/'rascar_v41'))
from common import B,CB,CL,DAYS,D,inputs,iou,baseline,metrics
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score

