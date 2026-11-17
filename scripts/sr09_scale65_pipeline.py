"""Scale65 rebuild -> calibration -> TCGA retrain -> paired/hires tests -> biology."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.nice(19)
from pathlib import Path
import sys,json,time,subprocess,inspect,shutil
import numpy as np,pandas as pd,cv2
from scipy.stats import spearmanr,rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));OUT=ROOT/'results/sr09/scale65_rebuild_2026-10-27';OUT.mkdir(exist_ok=True)
def status(phase,**kw):(OUT/'status.json').write_text(json.dumps(dict(phase=phase,pid=os.getpid(),**kw),indent=2)+'\n')
def run(script,log):
 with (OUT/log).open('a') as f:subprocess.run([sys.executable,'-u',script],cwd=ROOT,env=dict(os.environ,PROJECT_ROOT=str(ROOT)),stdout=f,stderr=subprocess.STDOUT,check=True)
status('waiting_fresh_scale65_fields')
while not(OUT/'prepare_complete.json').exists():time.sleep(10)
status('calibration_GSE327056_only')
if not(OUT/'calibration/calibration.json').exists():run('scripts/sr09_scale65_calibrate.py','calibration_process.log')
cal=json.loads((OUT/'calibration/calibration.json').read_text());lo,hi=cal['sigma_um_range']
import sr02_degrade_benchmark as original
source=inspect.getsource(original.degrade_real2);module='from sr02_degrade_benchmark import *\nSIGMA_RANGE='+repr((lo,hi))+'\n'+source.replace('rng.uniform(0.3, 1.2)','rng.uniform(*SIGMA_RANGE)');(ROOT/'scripts/sr09_scale65_degrade.py').write_text(module)
core=(ROOT/'scripts/sr08t_x8_core.py').read_text().replace('from sr02_degrade_benchmark import degrade_real2','from sr09_scale65_degrade import degrade_real2');(ROOT/'scripts/sr09_scale65_core.py').write_text(core)
cfg=json.loads((ROOT/'config/sr_x8_train.yaml').read_text());cfg.update(output_dir=str((OUT/'training').relative_to(ROOT)),protocol=str((OUT/'TRAINING_SPEC.md').relative_to(ROOT)),calibrated_blur_um=[lo,hi],GEO_physical_spot_um=65,GEO_neighbor_pitch_um=100,feature_circle_um=55);(OUT/'train_config.json').write_text(json.dumps(cfg,indent=2));(OUT/'TRAINING_SPEC.md').write_text('User2026-10-27 scale correction and retrain. GEO mpp=65/spot_diameter_fullres, hires mpp=native/sf; nearest neighbor100um cross-check. Feature circle55um. Only Gaussian range in SR08 training changes to'+str([lo,hi])+'um, calibrated withGSE3270564pairs. TCGA pixels/splits/official initialization/L1/15k steps/TCGA min validation L1 selection unchanged. GSE2118953donors held out; oldscale469step checkpoint not resumed.\n')
trainer=(ROOT/'scripts/sr08v_train_x8.py').read_text().replace('from sr08t_x8_core import','from sr09_scale65_core import').replace("config/sr_x8_train.yaml",str((OUT/'train_config.json').relative_to(ROOT))).replace('scripts/sr08t_x8_core.py','scripts/sr09_scale65_core.py').replace('python scripts/sr08v_train_x8.py','python scripts/sr09_scale65_train.py');(ROOT/'scripts/sr09_scale65_train.py').write_text(trainer)
(OUT/'code_at_launch').mkdir(exist_ok=True)
for f in ['sr09_scale65_train.py','sr09_scale65_core.py','sr09_scale65_degrade.py','sr09_scale65_prepare.py','sr09_scale65_calibrate.py','sr09_scale65_pipeline.py']:shutil.copy2(ROOT/'scripts'/f,OUT/'code_at_launch'/f)
status('training_scale65_blur',sigma_um_range=[lo,hi])
if not(OUT/'training/best.pth').exists():
 if (OUT/'training').exists():raise RuntimeError('Existing unfinished run retained. Explicit complete-checkpoint resume required; do not rewind.')
 run('scripts/sr09_scale65_train.py','training.log')
assert json.loads((OUT/'training/progress.json').read_text())['phase']=='complete'
import sr09_scale65_common as b
import sr09_scale65_biology as c
b.C=dict(b.C,weights=str((OUT/'training/best.pth').relative_to(ROOT)),weights_sha256=b.digest(OUT/'training/best.pth'));b.WORK=OUT/'eval_worklog';b.WORK.mkdir(exist_ok=True);c.OUT=OUT/'C';c.OUT.mkdir(exist_ok=True);c.offsets=lambda r,p,m:(np.zeros((len(m),2)),np.ones(2));audit=pd.read_csv(OUT/'scale_audit.csv');PRIMARY=audit[audit.heldout_test].gsm.tolist()
def aggregate(base,sid,methods):
 p=pd.read_csv(base/'patches.csv');tables={}
 for method in methods:
  n=b.nuclei(sid,method);g=n[np.hypot(n.x-255.5,n.y-255.5)<=110].groupby('patch_id').area_um2;t=pd.DataFrame(index=p.patch_id);t['n']=g.count().reindex(p.patch_id,fill_value=0);t['density']=t.n/(np.pi*.0275**2);t['area_median']=g.median().reindex(p.patch_id).where(t.n>=3);t['area_cv']=(g.std(ddof=0)/g.mean()).reindex(p.patch_id).where(t.n>=3);t.to_csv(base/f'spot_features_{method}.csv');tables[method]=t
 return tables
realmetrics=[]
for group,paired in [('paired',True),('hires',False)]:
 b.OUT=OUT/group
 for row in audit[audit.paired==paired].itertuples():
  sid=row.gsm;base=b.OUT/'samples'/sid;methods=['native','bicubic','sr'] if paired else ['bicubic','sr'];status('fresh_real_inference_segmentation',group=group,sample=sid);b.infer(sid)
  for method in methods:b.mosaic(sid,method);b.segment(sid,method)
  tables=aggregate(base,sid,methods)
  if sid in PRIMARY:
   ref=tables['native']
   for method in ['bicubic','sr']:
    for feat in ['density','area_median','area_cv']:
     x=ref[feat];y=tables[method][feat];eligible=np.ones(len(ref),bool) if feat=='density' else ref.n>=5;v=eligible&x.notna()&y.notna();bias=v&(x!=0);realmetrics.append(dict(gsm=sid,input='true_hires_scale65',method=method,feature=feat,rho=spearmanr(x[v],y[v]).statistic,median_absolute_relative_error=np.median(abs(y[bias]/x[bias]-1)),coverage=v.sum()/eligible.sum(),n_spots=len(ref),n_valid=int(v.sum())));
   pd.DataFrame(realmetrics).to_csv(OUT/'real_metrics.csv',index=False)
  status('C_features_and_expression',group=group,sample=sid);dest,met=c.features(sid,base,row if paired else None);(dest/'image_sources.json').write_text(json.dumps(dict(hires=row.hires_path,sf=row.tissue_hires_scalef)));c.correlate(sid,base,dest,met,row.expression_folder,'A_paired' if paired else 'B_hires')
 c.summary('A_paired' if paired else 'B_hires')
# Same newly cut fields generate simulated calibrated-v2 inputs; no old-scale patches reused.
from sr09_scale65_degrade import degrade_real2
simmetrics=[];b.OUT=OUT/'sim'
for si,sid in enumerate(PRIMARY):
 src=OUT/'paired/samples'/sid;dst=b.OUT/'samples'/sid;dst.mkdir(parents=True,exist_ok=True);p=pd.read_csv(src/'patches.csv')
 if not(dst/'prepared.json').exists():
  rows=[]
  for i,r in enumerate(p.itertuples()):
   if i%100==0:b.gate();status('fresh_simulated_inputs',sample=sid,done=i,total=len(p))
   gt=cv2.resize(b.rgb(src/'patches/native'/f'{r.patch_id}.png'),(256,256),interpolation=cv2.INTER_AREA);lr,params=degrade_real2(gt,32,np.random.default_rng([20261027,si,i]));lp=dst/'patches/lr'/f'{r.patch_id}.png';bp=dst/'patches/bicubic'/f'{r.patch_id}.png'
   if not lp.exists():b.png(lp,lr)
   if not bp.exists():b.png(bp,cv2.resize(cv2.resize(lr,(256,256),interpolation=cv2.INTER_CUBIC),(512,512),interpolation=cv2.INTER_CUBIC))
   rows.append(dict(patch_id=r.patch_id,barcode=r.barcode,fullres_x=r.fullres_x,fullres_y=r.fullres_y,degradation_params=json.dumps(params),lr_sha256=b.digest(lp)))
  pd.DataFrame(rows).to_csv(dst/'patches.csv',index=False);(dst/'prepared.json').write_text(json.dumps(dict(scale65=True,field_um=128,GT_mpp=.5,LR_mpp=4,Gaussian_range=[lo,hi])))
 b.infer(sid)
 for method in ['bicubic','sr']:b.mosaic(sid,method);b.segment(sid,method)
 tables=aggregate(dst,sid,['bicubic','sr']);ref=pd.read_csv(src/'spot_features_native.csv').set_index('patch_id')
 for method in tables:
  for feat in ['density','area_median','area_cv']:
   x=ref[feat];y=tables[method][feat];eligible=np.ones(len(ref),bool) if feat=='density' else ref.n>=5;v=eligible&x.notna()&y.notna();bias=v&(x!=0);simmetrics.append(dict(gsm=sid,input='simulated_calibrated_v2_scale65',method=method,feature=feat,rho=spearmanr(x[v],y[v]).statistic,median_absolute_relative_error=np.median(abs(y[bias]/x[bias]-1)),coverage=v.sum()/eligible.sum(),n_spots=len(ref),n_valid=int(v.sum())))
 pd.DataFrame(simmetrics).to_csv(OUT/'simulation_metrics.csv',index=False)
combined=pd.DataFrame(realmetrics+simmetrics);combined.to_csv(OUT/'comparison_long.csv',index=False);wide=combined.pivot(index=['gsm','method','feature'],columns='input',values=['rho','median_absolute_relative_error','coverage']);wide.columns=['__'.join(x) for x in wide.columns];wide.to_csv(OUT/'comparison_side_by_side.csv')
# Epithelial masks are expression-only; preserve selection from the previous round.
rows=[];legacy=ROOT/'results/sr_c_explore_2026-10-26_legacy_scale55/round2_tumor'
for group in ['A_paired','B_hires']:
 for folder in (OUT/'C'/group).iterdir():
  if not(folder/'spot_expression_sr.csv').exists():continue
  selection=pd.read_csv(legacy/f'{folder.name}_selection.csv');ids=set(selection.loc[selection.selected,'barcode'])
  for method in (['native','sr','bicubic'] if group=='A_paired' else ['sr','bicubic']):
   z=pd.read_csv(folder/f'spot_expression_{method}.csv').set_index('barcode');z=z[z.index.isin(ids)];z['basal_minus_classical']=z.basal-z.classical
   for feat in ['area_median','area_cv','eccentricity_median','h_mean']:
    for prog in ['basal','classical','basal_minus_classical','proliferation','hypoxia']:
     v=z[[feat,prog,'total_umi','density']].dropna();A=np.column_stack([np.ones(len(v)),rankdata(v.total_umi),rankdata(v.density)]);x=rankdata(v[feat]);y=rankdata(v[prog]);x-=A@np.linalg.lstsq(A,x,rcond=None)[0];y-=A@np.linalg.lstsq(A,y,rcond=None)[0];rows.append(dict(group=group,gsm=folder.name,method=method,feature=feat,program=prog,n=len(v),rho=spearmanr(v[feat],v[prog]).statistic if len(v)>2 else np.nan,partial_rho=np.corrcoef(x,y)[0,1] if len(v)>2 else np.nan))
pd.DataFrame(rows).to_csv(OUT/'C/round2_tumor_associations.csv',index=False);status('complete',all_fresh_scale65=True,comparison_rows=len(wide),C_complete=True)
