"""Correct three hires scales and compare locked old/new SR models at fixed scale."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.nice(19)
from pathlib import Path
import sys,json,shutil,ast,datetime,subprocess
import numpy as np,pandas as pd,cv2
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import sr09_scale65_common as b
import sr09_scale65_biology as c
from sr09_scale65_prepare import read,positions
BASE=R/'results/sr09/scale65_rebuild_2026-10-27';OUT=R/'results/sr09/pitch_ablation_2026-10-28';OUT.mkdir(exist_ok=True);cv2.setNumThreads(2)
def status(phase,**kw):(OUT/'status.json').write_text(json.dumps(dict(phase=phase,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2))
def link(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():dst.symlink_to(src)
# Reuse the exact aggregation definition without executing the old pipeline.
src=(R/'scripts/sr09_scale65_pipeline.py').read_text();node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='aggregate');exec(ast.get_source_segment(src,node))
audit=pd.read_csv(BASE/'scale_audit.csv');fix=['GSM9861745','GSM9051173','GSM8454387'];audit['scale_basis']='65um diameter'
for sid in fix:
 ix=audit.index[audit.gsm==sid][0];m=100/audit.loc[ix,'neighbor_pitch_fullres_px'];audit.loc[ix,'native_mpp']=m;audit.loc[ix,'hires_mpp']=m/audit.loc[ix,'tissue_hires_scalef'];audit.loc[ix,'neighbor_pitch_um']=100.;audit.loc[ix,'scale_basis']='100um median neighbor pitch'
audit.to_csv(OUT/'scale_audit.csv',index=False);b.WORK=OUT/'worklog';b.WORK.mkdir(exist_ok=True);c.OUT=OUT/'C';c.OUT.mkdir(exist_ok=True);c.offsets=lambda r,p,m:(np.zeros((len(m),2)),np.ones(2))
b.C=dict(b.C,weights=str(BASE/'training/best.pth'),weights_sha256=b.digest(BASE/'training/best.pth'));b.OUT=OUT/'corrected_hires'
for sid in fix:
 row=audit.set_index('gsm').loc[sid];dst=b.OUT/'samples'/sid;dst.mkdir(parents=True,exist_ok=True);status('recrop_pitch_scale',sample=sid)
 if not(dst/'prepared.json').exists():
  p=positions(row.positions_path);im=cv2.imdecode(np.frombuffer(read(row.hires_path),np.uint8),cv2.IMREAD_COLOR)[...,::-1];records=[];sf=row.tissue_hires_scalef
  for i,z in enumerate(p.itertuples()):
   if i%100==0:b.gate()
   pid=f'p{i:05d}';lr=b.crop_at(im,(z.x+.5)*sf-.5,(z.y+.5)*sf-.5,32,4/row.hires_mpp,4/row.hires_mpp);lp=dst/'patches/lr'/f'{pid}.png';bp=dst/'patches/bicubic'/f'{pid}.png'
   if not lp.exists():b.png(lp,lr)
   if not bp.exists():b.png(bp,cv2.resize(cv2.resize(lr,(256,256),interpolation=cv2.INTER_CUBIC),(512,512),interpolation=cv2.INTER_CUBIC))
   records.append(dict(patch_id=pid,barcode=z.barcode,fullres_x=z.x,fullres_y=z.y,field_um=128,lr_sha256=b.digest(lp)))
  pd.DataFrame(records).to_csv(dst/'patches.csv',index=False);(dst/'prepared.json').write_text(json.dumps(dict(native_mpp=row.native_mpp,hires_mpp=row.hires_mpp,hires_scalef=sf,feature_circle_um=55,field_um=128,scale_basis='100um median neighbor pitch')))
 status('corrected_hires_inference_segmentation',sample=sid);b.infer(sid)
 for method in ['bicubic','sr']:b.mosaic(sid,method);b.segment(sid,method)
 aggregate(dst,sid,['bicubic','sr']);dest,met=c.features(sid,dst,None);(dest/'image_sources.json').write_text(json.dumps(dict(hires=row.hires_path,sf=row.tissue_hires_scalef)));c.correlate(sid,dst,dest,met,row.expression_folder,'B_hires')
# Unchanged samples are referenced read-only, not reanalysed with altered image scales.
for group in ['A_paired','B_hires']:
 for folder in (BASE/'C'/group).iterdir():
  if folder.is_dir() and folder.name not in fix:link(folder,c.OUT/group/folder.name)
for group in ['A_paired','B_hires']:c.summary(group)
# Original SR08 model on the identical already-correct 65um donor fields.
old=R/'results/sr08/models/x8/2026-10-24_run01/best.pth';b.C.update(weights=str(old),weights_sha256=b.digest(old));b.OUT=OUT/'old_SR08_correct_scale';rows=[]
for sid in ['GSM6505133','GSM6505134','GSM6505135']:
 status('old_model_correct_scale',sample=sid);src=BASE/'paired/samples'/sid;dst=b.OUT/'samples'/sid;dst.mkdir(parents=True,exist_ok=True)
 for name in ['patches.csv','prepared.json']:link(src/name,dst/name)
 link(src/'patches/lr',dst/'patches/lr');b.infer(sid);b.mosaic(sid,'sr');b.segment(sid,'sr');tables=aggregate(dst,sid,['sr']);ref=pd.read_csv(src/'spot_features_native.csv').set_index('patch_id')
 for feat in ['density','area_median','area_cv']:
  x=ref[feat];y=tables['sr'][feat];eligible=np.ones(len(ref),bool) if feat=='density' else ref.n>=5;v=eligible&x.notna()&y.notna();bias=v&(x!=0);rows.append(dict(gsm=sid,method='old_SR08',feature=feat,rho=spearmanr(x[v],y[v]).statistic,median_absolute_relative_error=np.median(abs(y[bias]/x[bias]-1)),coverage=v.sum()/eligible.sum(),n_valid=int(v.sum())))
 pd.DataFrame(rows).to_csv(OUT/'old_model_metrics.csv',index=False)
real=pd.read_csv(BASE/'real_metrics.csv');real['method']=real.method.replace({'sr':'calibrated_SR','bicubic':'bicubic'});combined=pd.concat([real,pd.DataFrame(rows)],ignore_index=True);combined.to_csv(OUT/'model_ablation_long.csv',index=False);combined.pivot(index=['gsm','feature'],columns='method',values=['rho','median_absolute_relative_error','coverage']).to_csv(OUT/'model_ablation_side_by_side.csv')
# Recreate both C summaries and tumor-enriched displays using corrected features.
for f in c.OUT.iterdir():
 if f.suffix in ['.mapng','.mapdf']:f.rename(f.with_suffix('.png' if f.suffix=='.mapng' else '.pdf'))
s=(R/'scripts/sr09_scale65_round2_figures.py').read_text().replace("BASE=ROOT/'results/sr09/scale65_rebuild_2026-10-27/C'","BASE=ROOT/'results/sr09/pitch_ablation_2026-10-28/C'")
s=s.replace("base=BASE.parent/'paired/samples'/sid if g=='A_paired' else BASE.parent/'hires/samples'/sid","base=ROOT/'results/sr09/scale65_rebuild_2026-10-27/paired/samples'/sid if g=='A_paired' else (BASE.parent/'corrected_hires/samples'/sid if sid in ['GSM9861745','GSM9051173','GSM8454387'] else ROOT/'results/sr09/scale65_rebuild_2026-10-27/hires/samples'/sid)")
p=R/'scripts/sr09_pitch_round2_figures.py';p.write_text(s)
with (OUT/'round2.log').open('a') as f:subprocess.run([sys.executable,str(p)],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
status('compute_complete_report_pending')
