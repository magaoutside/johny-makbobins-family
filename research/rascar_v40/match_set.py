"""Use the frozen V39 matching algorithm, with V40 prediction/output paths."""
from common40 import *
import matching,argparse

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-run',required=True);p.add_argument('--full-model-run');args=p.parse_args()
    matching.A=A
    matching.collect_evidence(args.model_run,args.model_run+'_identity',identity=True,full_model_run=args.full_model_run)
