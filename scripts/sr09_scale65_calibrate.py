"""Calibrate Gaussian blur on GSE327056 only; held-out donors never read."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.nice(19)
from pathlib import Path
import sys,json,inspect,time,hashlib
import cv2,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));import sr02_degrade_benchmark as old
OUT=ROOT/'results/sr09/scale65_rebuild_2026-10-27/calibration';OUT.mkdir(exist_ok=True);cv2.setNumThreads(2)
source=inspect.getsource(old.degrade_real2);assert 'rng.uniform(0.3, 1.2)' in source
namespace=vars(old).copy();exec(source.replace('def degrade_real2(', 'def degrade_sigma(').replace('rng.uniform(0.3, 1.2)','rng.uniform(SIGMA, SIGMA)'),namespace);fn=namespace['degrade_sigma']
grid=np.array([.3,.5,.75,1.,1.25,1.5,1.75,2.,2.25,2.5,3.,4.,6.,8.,12.,16.,24.]);audit=pd.read_csv(ROOT/'results/sr09/scale65_rebuild_2026-10-27/scale_audit.csv');audit=audit[audit.gse=='GSE327056'];assert len(audit)==4;rows=[]
def measures(im):
 g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;lap=cv2.Laplacian(g,cv2.CV_32F).var();z=g-g.mean();power=np.abs(np.fft.fft2(z))**2/g.size**2;fy,fx=np.meshgrid(np.fft.fftfreq(g.shape[0]),np.fft.fftfreq(g.shape[1]),indexing='ij');radius=np.hypot(fx,fy);bands=np.array([power[(radius>=a)&(radius<b)].mean() for a,b in zip([.03125,.125,.25,.375,.5],[.125,.25,.375,.5,.72])]);return np.r_[lap,bands]
for si,r in enumerate(audit.itertuples()):
 base=ROOT/'results/sr09/scale65_rebuild_2026-10-27/paired/samples'/r.gsm;p=pd.read_csv(base/'patches.csv');indices=np.linspace(0,len(p)-1,32,dtype=int)
 for i in indices:
  while os.getloadavg()[0]>8:time.sleep(10)
  pid=p.patch_id.iloc[i];native=cv2.imread(str(base/'patches/native'/f'{pid}.png'))[...,::-1];gt=cv2.resize(native,(256,256),interpolation=cv2.INTER_AREA);target=cv2.imread(str(base/'patches/lr'/f'{pid}.png'))[...,::-1];tm=measures(target)
  for sigma in grid:
   namespace['SIGMA']=float(sigma);im,prm=fn(gt,32,np.random.default_rng([20261027,si,int(i)]));m=measures(im);logratio=np.log((m+1e-10)/(tm+1e-10));loss=.5*logratio[0]**2+.5*np.mean(logratio[1:]**2)
   rows.append(dict(gsm=r.gsm,patch_id=pid,sigma_um=sigma,loss=loss,lapvar_sim=m[0],lapvar_real=tm[0],**{f'psd_sim_{j}':m[j+1] for j in range(5)},**{f'psd_real_{j}':tm[j+1] for j in range(5)}))
  print(r.gsm,pid,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'calibration_grid.csv',index=False)
t=pd.DataFrame(rows);idx=t.groupby(['gsm','patch_id']).loss.idxmin();best=t.loc[idx].copy();best.to_csv(OUT/'calibration_patch_optima.csv',index=False);lo,hi=np.quantile(best.sigma_um,[.1,.9]);
if hi<=lo:lo=max(.05,lo*.8);hi=hi*1.2
summary=t.groupby(['gsm','sigma_um']).loss.median().reset_index();summary.to_csv(OUT/'calibration_sample_curves.csv',index=False);record=dict(sigma_um_range=[float(lo),float(hi)],grid_um=grid.tolist(),n_fields=128,calibration_samples=audit.gsm.tolist(),held_out_samples=['GSM6505133','GSM6505134','GSM6505135'],criterion='0.5 squared log Laplacian ratio +0.5 mean squared log PSD-band ratios, unchanged v2 stain/JPEG/defocus/noise for each fixed seed',range_definition='Per-field best grid sigma10th-90th percentiles; if degenerate expand center+/-20%; empirical training range, not confidence interval',boundary_fraction=float(best.sigma_um.isin([grid[0],grid[-1]]).mean()),seed=20261027,source_degrade_sha256=hashlib.sha256(inspect.getsource(old.degrade_real2).encode()).hexdigest(),note='Only Gaussian range is calibrated; retained noise/stain/JPEG may prevent simultaneous exact PSD/Laplacian matching.')
(OUT/'calibration.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
for sid,z in summary.groupby('gsm'):ax.plot(z.sigma_um,z.loss,'o-',label=sid)
ax.axvspan(lo,hi,color='gray',alpha=.15);ax.set_xscale('log');ax.set_xlabel('Gaussian sigma (um, applied at0.5um/px)');ax.set_ylabel('Median spectral/Laplacian discrepancy');ax.legend(fontsize=8)
for ext in ['png','pdf']:fig.savefig(OUT/f'calibration.{ext}',dpi=160)
