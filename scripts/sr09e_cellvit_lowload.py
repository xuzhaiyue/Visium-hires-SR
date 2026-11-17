"""Reuse the existing MPS wrapper, adding a pause/load gate once per GPU batch."""
import os,time,sys
os.environ.setdefault('CELLVIT_TORCH_THREADS','2')
import run_cellvit_mps as base
from pathlib import Path
from cellvit.utils import cache_models
def cached_only(directory_path,file_name,**kwargs):
    path=Path(directory_path)/file_name
    if not path.is_file():
        raise FileNotFoundError(f'Downloads are forbidden for SR09-B: missing cached {path}')
cache_models.check_and_download=cached_only
os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
original=base.INF.CellViTInference.apply_softmax_reorder
calls=0
def throttled(self,predictions):
    global calls
    result=original(self,predictions)
    time.sleep(.5)
    while os.getloadavg()[0]>8:time.sleep(10)
    calls+=1
    if calls%50==0:print(f'SR09 low-load batches={calls}, nice={os.getpriority(os.PRIO_PROCESS,0)}, load={os.getloadavg()[0]:.2f}',flush=True)
    return result
base.INF.CellViTInference.apply_softmax_reorder=throttled
if __name__=='__main__':sys.exit(base.detect_cells.main())
