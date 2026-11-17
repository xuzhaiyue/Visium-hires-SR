"""Run locked SwinIR inference in the existing hemorph environment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
import argparse,json
from pathlib import Path
import sr09_scale65_common as b
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--work',required=True);p.add_argument('--sample',required=True);p.add_argument('--weights',required=True);p.add_argument('--sha256',required=True);a=p.parse_args()
b.OUT=Path(a.output);b.WORK=Path(a.work);b.WORK.mkdir(parents=True,exist_ok=True)
assert b.digest(a.weights)==a.sha256
b.C.update(weights=a.weights,weights_sha256=a.sha256,batch=4,pause_seconds=.5,load_threshold=8)
b.infer(a.sample)
z=json.loads((b.OUT/'samples'/a.sample/'inference_complete.json').read_text());assert z['weights_sha256']==a.sha256
