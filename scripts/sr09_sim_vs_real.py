"""Paired simulated-v2 and true-hires spot concordance with locked SR8."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import sys,json,datetime
import cv2,numpy as np,pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));import sr09d_run_b as b
from sr02_degrade_benchmark import degrade_real2
OUT=ROOT/'results/sr09/B_sim_vs_real_2026-10-27';OUT.mkdir(parents=True,exist_ok=True);b.OUT=OUT;b.WORK=OUT/'worklog';b.WORK.mkdir(exist_ok=True);cv2.setNumThreads(2)
assert b.digest(ROOT/b.C['weights'])==b.C['weights_sha256'];PRIMARY=['GSM6505133','GSM6505134','GSM6505135'];OLD=ROOT/'results/sr09/B_2026-10-25/samples';CORRECTED=ROOT/'results/sr_c_explore_2026-10-26/B_coordinate_recomputed';SEED=20261027
config=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),samples=PRIMARY,seed=SEED,degradation='scripts/sr02_degrade_benchmark.py:degrade_real2; v2 unchanged',GT='existing native 512px@.25 -> INTER_AREA256px@.5',LR='32px@4um',weights=b.C['weights'],weights_sha256=b.C['weights_sha256'],threads=2,nice=19,sr_batch=4,cellvit_batch=2,ray_workers=1,pause=.5,load_wait=8,no_training=True,no_download=True,reference=str(CORRECTED),spot_center_model=255.5)
if not (OUT/'run_config.json').exists():(OUT/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
allrows=[]
for sample_idx,sid in enumerate(PRIMARY):
 dst=OUT/'samples'/sid;dst.mkdir(parents=True,exist_ok=True);src=OLD/sid;p=pd.read_csv(src/'patches.csv');print(sid,flush=True)
 if not (dst/'prepared.json').exists():
  rows=[]
  for i,r in enumerate(p.itertuples()):
   if i%100==0:b.gate();b.status('simulating',sample=sid,done=i,total=len(p))
   path=src/'patches/native'/f'{r.patch_id}.png';gt=cv2.resize(b.rgb(path),(256,256),interpolation=cv2.INTER_AREA);rng=np.random.default_rng([SEED,sample_idx,i]);lr,params=degrade_real2(gt,32,rng)
   lp=dst/'patches/lr'/f'{r.patch_id}.png';bp=dst/'patches/bicubic'/f'{r.patch_id}.png'
   if not lp.exists():b.png(lp,lr)
   if not bp.exists():b.png(bp,cv2.resize(cv2.resize(lr,(256,256),interpolation=cv2.INTER_CUBIC),(512,512),interpolation=cv2.INTER_CUBIC))
   rows.append(dict(patch_id=r.patch_id,barcode=r.barcode,fullres_x=r.fullres_x,fullres_y=r.fullres_y,source_native=str(path),native_sha256=b.digest(path),lr_sha256=b.digest(lp),degradation_params=json.dumps(params),seed_components=json.dumps([SEED,sample_idx,i])))
  pd.DataFrame(rows).to_csv(dst/'patches.csv',index=False);b.atomic_json(dst/'prepared.json',dict(n_patches=len(p),same_spots_as=str(src/'patches.csv'),field_um=128,GT_mpp=.5,LR_mpp=4))
 b.infer(sid)
 for method in ['bicubic','sr']:b.mosaic(sid,method);b.segment(sid,method)
 ref=pd.read_csv(CORRECTED/sid/'spot_features_native.csv').set_index('patch_id').reindex(p.patch_id)
 for method in ['bicubic','sr']:
  n=b.nuclei(sid,method);g=n[np.hypot(n.x-255.5,n.y-255.5)<=110].groupby('patch_id').area_um2;t=pd.DataFrame(index=p.patch_id);t['n']=g.count().reindex(p.patch_id,fill_value=0);t['density']=t.n/(np.pi*.0275**2);t['area_median']=g.median().reindex(p.patch_id).where(t.n>=3);t['area_cv']=(g.std(ddof=0)/g.mean()).reindex(p.patch_id).where(t.n>=3);t.to_csv(dst/f'spot_features_{method}.csv')
  for feat in ['density','area_median','area_cv']:
   eligible=np.ones(len(ref),bool) if feat=='density' else ref.n>=5;x=ref[feat];y=t[feat];v=eligible&x.notna()&y.notna();bias=v&(x!=0)
   allrows.append(dict(gsm=sid,input='simulated_real2_4.0',method=method,feature=feat,n_spots=len(p),n_reference_eligible=int(eligible.sum()),n_valid=int(v.sum()),coverage=float(v.sum()/eligible.sum()),rho=float(spearmanr(x[v],y[v]).statistic),median_absolute_relative_error=float(np.median(abs(y[bias]/x[bias]-1)))))
 pd.DataFrame(allrows).to_csv(OUT/'simulation_metrics.csv',index=False)
real=pd.read_csv(CORRECTED/'metrics_by_sample.csv');real=real[real.gsm.isin(PRIMARY)].copy();real['input']='true_hires';combined=pd.concat([real,pd.DataFrame(allrows)],ignore_index=True);combined.to_csv(OUT/'metrics_long.csv',index=False)
wide=combined.pivot(index=['gsm','method','feature'],columns='input',values=['rho','median_absolute_relative_error','coverage']);wide.columns=['__'.join(c) for c in wide.columns];wide.to_csv(OUT/'comparison_side_by_side.csv');b.status('simulation_comparison_complete');print(wide,flush=True)
