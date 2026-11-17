"""Serial locked-model HEST FFPE resampling-bracket experiment; no downloads."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
from pathlib import Path
import sys,json,time,inspect,traceback,subprocess,datetime,shutil
import numpy as np,pandas as pd,cv2,tifffile
ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]));sys.path.insert(0,str(ROOT/'scripts'))
import sr09_scale65_common as b
import sr09_scale65_biology as c
from sr09_reproduce_hires_candidate import reproduce_hires
OUT=ROOT/'results/sr09/hest_bounds_2026-10-29';BASE=ROOT/'results/sr09/scale65_rebuild_2026-10-27'
NEW=BASE/'training/best.pth';OLD=ROOT/'results/sr08/models/x8/2026-10-24_run01/best.pth'
CONDITIONS={'upper_area':'opencv_area','lower_aa0':'skimage_aa0'}
cv2.setNumThreads(2)
def crop_at(im,cx,cy,n,sx,sy):
 # Subtract large native origins before casting maps to OpenCV float32.
 x=cx+(np.arange(n,dtype=np.float64)+.5-n/2)*sx;y=cy+(np.arange(n,dtype=np.float64)+.5-n/2)*sy
 x0=max(0,int(np.floor(x.min()))-4);y0=max(0,int(np.floor(y.min()))-4);x1=min(im.shape[1],int(np.ceil(x.max()))+5);y1=min(im.shape[0],int(np.ceil(y.max()))+5)
 if x1<=x0 or y1<=y0:return np.full((n,n,3),255,np.uint8)
 xx,yy=np.meshgrid((x-x0).astype(np.float32),(y-y0).astype(np.float32))
 return cv2.remap(im[y0:y1,x0:x1],xx,yy,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
b.crop_at=crop_at
def infer_in_existing_hemorph(sid):
 # Keep resampling/CellViT in cellvit; timm and SwinIR already exist in hemorph.
 args=[str(Path.home()/'miniforge3/envs/hemorph/bin/python'),'-u',str(ROOT/'scripts/sr09_hest_bounds_infer.py'),'--output',str(b.OUT),'--work',str(b.WORK),'--sample',sid,'--weights',str(b.C['weights']),'--sha256',b.C['weights_sha256']]
 subprocess.run(args,cwd=ROOT,check=True)
b.infer=infer_in_existing_hemorph
def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2));os.replace(temp,path)
def status(phase,**kw):
 z=dict(phase=phase,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw);atomic(OUT/'status.json',z)
 with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(z)+'\n')
 print(z,flush=True)
def gate():
 while os.getloadavg()[0]>8:status('waiting_low_load',load=os.getloadavg()[0]);time.sleep(10)
def link(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():dst.symlink_to(src)
def setup_features():
 text=inspect.getsource(c.features).replace("methods=['native','bicubic','sr'] if r is not None else ['bicubic','sr']","methods=p['feature_methods']")
 text=text.replace("if i%200==0:print(i,'/',len(manifest),flush=True)","if i%100==0:\n    while os.getloadavg()[0]>8:time.sleep(10)\n    print(i,'/',len(manifest),flush=True);time.sleep(.1)")
 exec(text,c.__dict__);c.offsets=lambda r,p,m:(np.zeros((len(m),2)),np.ones(2))
def sample_path(condition,sid):return OUT/condition/'samples'/sid

def prepare(r):
 sid=r['sample'];ref=sample_path('reference',sid);ref.mkdir(parents=True,exist_ok=True)
 if (ref/'prepare_complete.json').exists() and all((sample_path(k,sid)/'prepare_complete.json').exists() for k in CONDITIONS):return
 gate();status('prepare_native_and_hires',sample=sid)
 with tifffile.TiffFile(r['native_path']) as tf:native=tf.pages[0].asarray(maxworkers=2)
 if native.shape[0] in (3,4) and native.shape[-1] not in (3,4):native=np.moveaxis(native,0,-1)
 native=native[...,:3];assert native.dtype==np.uint8 and native.shape[:2]==(r['native_height'],r['native_width'])
 p=pd.read_csv(OUT/'inputs'/sid/'positions.csv');p=p[p.image_eligible].reset_index(drop=True);p['patch_id']=[f'p{i:05d}' for i in range(len(p))];p['fullres_x']=p.x;p['fullres_y']=p.y;p['field_um']=128
 if not(ref/'patches.csv').exists():p.to_csv(ref/'patches.csv',index=False)
 for i,z in enumerate(p.itertuples()):
  if i%100==0:gate();status('native_patches',sample=sid,done=i,total=len(p))
  path=ref/'patches/native'/f'{z.patch_id}.png'
  if not path.exists():b.png(path,b.crop_at(native,z.x,z.y,512,.25/r['native_mpp'],.25/r['native_mpp']))
 atomic(ref/'prepared.json',dict(feature_methods=['native'],native_mpp=r['native_mpp'],field_um=128,feature_circle_um=55,center=255.5))
 atomic(ref/'prepare_complete.json',dict(patches=len(p),image_coverage_mask=str(OUT/'inputs'/sid/'positions.csv')))
 for condition,method in CONDITIONS.items():
  dst=sample_path(condition,sid);dst.mkdir(parents=True,exist_ok=True)
  if (dst/'prepare_complete.json').exists():continue
  hp=dst/'generated_hires.png'
  if hp.exists():hi=b.rgb(hp)
  else:
   gate();status('generate_hires',sample=sid,condition=condition,resampler=method);hi=reproduce_hires(native,float(r['hires_scalef']),method=method);b.png(hp,hi)
  sx=hi.shape[1]/native.shape[1];sy=hi.shape[0]/native.shape[0];mx=r['native_mpp']/sx;my=r['native_mpp']/sy
  link(ref/'patches/native',dst/'patches/native');link(ref/'patches.csv',dst/'patches.csv')
  for i,z in enumerate(p.itertuples()):
   if i%100==0:gate();status('generated_lr_patches',sample=sid,condition=condition,done=i,total=len(p))
   lp=dst/'patches/lr'/f'{z.patch_id}.png';bp=dst/'patches/bicubic'/f'{z.patch_id}.png'
   if not lp.exists():b.png(lp,b.crop_at(hi,(z.x+.5)*sx-.5,(z.y+.5)*sy-.5,32,4/mx,4/my))
   if not bp.exists():lr=b.rgb(lp);b.png(bp,cv2.resize(cv2.resize(lr,(256,256),interpolation=cv2.INTER_CUBIC),(512,512),interpolation=cv2.INTER_CUBIC))
  atomic(dst/'prepared.json',dict(feature_methods=['native','bicubic','sr'],resampler=method,generated_not_measured=True,native_mpp=r['native_mpp'],nominal_hires_mpp=r['hires_mpp'],actual_hires_mpp_xy=[mx,my],hires_scalef_nominal=r['hires_scalef'],actual_scalef_xy=[sx,sy],hires_shape=list(hi.shape),field_um=128,model_input_mpp=4,model_input_wh=32,feature_circle_um=55,center=255.5,mapping='(x+.5)*actual_scale-.5; physical128um; same center all methods',generated_hires_sha256=b.digest(hp)))
  atomic(dst/'prepare_complete.json',dict(patches=len(p),resampler=method));del hi
 del native

def segment(root,sid,method):
 b.OUT=OUT/root;status('mosaic_and_cellvit',sample=sid,root=root,method=method);gate();b.mosaic(sid,method);b.segment(sid,method)
 marker=sample_path(root,sid)/'mosaic'/method/'cellvit_complete.json';z=json.loads(marker.read_text());assert Path(z['cells_path']).is_file() and b.digest(z['cells_path'])==z['cells_sha256']

def features(root,sid,reference=None):
 dst=sample_path(root,sid);c.OUT=OUT/'biology'/root;c.OUT.mkdir(parents=True,exist_ok=True);dest=c.OUT/'A_paired'/sid;dest.mkdir(parents=True,exist_ok=True)
 if reference:link(reference/'features_native.csv',dest/'features_native.csv')
 if not(dest/'features_complete.json').exists():
  status('morphology_features',sample=sid,root=root);gate();d,methods=c.features(sid,dst,object());atomic(dest/'features_complete.json',dict(methods=methods,feature_files={m:b.digest(d/f'features_{m}.csv') for m in methods}))
 return dest

def correlations(root,sid,methods,dest):
 marker=dest/'correlations_complete.json'
 if marker.exists():return
 scores=pd.read_csv(OUT/'inputs'/sid/'program_scores.csv',index_col=0);cov=pd.read_csv(OUT/'inputs'/sid/'gene_coverage.csv').to_dict('records');r=AUDIT.set_index('sample').loc[sid]
 c.expression=lambda folder:(scores,cov,[r.expression_path]);status('expression_associations',sample=sid,root=root)
 c.correlate(sid,sample_path(root,sid),dest,methods,None,'HEST_FFPE');atomic(marker,dict(associations_sha256=b.digest(dest/'associations.csv')))

def run_sample(r):
 sid=r['sample'];prepare(r);segment('reference',sid,'native');ref=features('reference',sid)
 for condition in CONDITIONS:
  root=condition;dst=sample_path(root,sid);b.OUT=OUT/root;b.C.update(weights=str(NEW),weights_sha256=MODEL_HASHES['calibrated_SR']);status('locked_new_SR',sample=sid,condition=condition);gate()
  if (dst/'inference_complete.json').exists():assert json.loads((dst/'inference_complete.json').read_text())['weights_sha256']==MODEL_HASHES['calibrated_SR']
  b.infer(sid)
  for method in ('bicubic','sr'):segment(root,sid,method)
  dest=features(root,sid,ref);correlations(root,sid,['native','bicubic','sr'],dest)
  oldroot=condition+'_old_SR08';old=sample_path(oldroot,sid);old.mkdir(parents=True,exist_ok=True)
  for name in ('patches.csv',):link(dst/name,old/name)
  link(dst/'patches/lr',old/'patches/lr');atomic(old/'prepared.json',dict(feature_methods=['sr'],field_um=128,feature_circle_um=55,center=255.5))
  b.OUT=OUT/oldroot;b.C.update(weights=str(OLD),weights_sha256=MODEL_HASHES['old_SR08']);status('locked_old_SR',sample=sid,condition=condition);gate()
  if (old/'inference_complete.json').exists():assert json.loads((old/'inference_complete.json').read_text())['weights_sha256']==MODEL_HASHES['old_SR08']
  b.infer(sid);segment(oldroot,sid,'sr');od=features(oldroot,sid);correlations(oldroot,sid,['sr'],od)
 atomic(OUT/'sample_complete'/f'{sid}.json',dict(conditions=7,feature_circle_um=55));status('sample_compute_complete',sample=sid)

def main():
 global AUDIT,MODEL_HASHES
 assert (OUT/'preflight_complete.json').exists();AUDIT=pd.read_csv(OUT/'sample_audit.csv');assert len(AUDIT)==9
 b.WORK=OUT/'common_status';b.WORK.mkdir(exist_ok=True);MODEL_HASHES={'calibrated_SR':b.digest(NEW),'old_SR08':b.digest(OLD)}
 codefiles=[Path(__file__),ROOT/'scripts/sr09_scale65_common.py',ROOT/'scripts/sr09_scale65_biology.py',ROOT/'scripts/sr09_reproduce_hires_candidate.py',ROOT/'scripts/sr09e_cellvit_lowload.py']
 cfg=dict(date='2026-10-29',source_audit=str(OUT/'sample_audit.csv'),model_paths=dict(calibrated_SR=str(NEW),old_SR08=str(OLD)),model_sha256=MODEL_HASHES,conditions=CONDITIONS,native_reference_conditions=9,generated_conditions=54,total_cellvit_conditions=63,seed=2026,nice=19,threads=2,SR_batch=4,CellViT_batch=2,pause_seconds=.5,load_pause=8,model_input_mpp=4,native_and_model_output_mpp=.25,field_um=128,feature_circle_um=55,matrix_features=c.FEATURES,programs=c.PROGRAMS,no_download=True,no_training=True,code_hashes={str(p):b.digest(p) for p in codefiles})
 if not(OUT/'run_config.json').exists():atomic(OUT/'run_config.json',cfg)
 else:assert json.loads((OUT/'run_config.json').read_text())['model_sha256']==MODEL_HASHES
 setup_features();atomic(OUT/'process.json',dict(pid=os.getpid(),command='cellvit python scripts/sr09_hest_bounds_pipeline.py',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));status('running',samples=9,cellvit_conditions=63)
 for r in AUDIT.to_dict('records'):run_sample(r)
 status('summarizing');subprocess.run([sys.executable,str(ROOT/'scripts/sr09_hest_bounds_report.py')],cwd=ROOT,check=True)
 status('complete',samples=9,cellvit_conditions=63,report='README.md')
if __name__=='__main__':
 try:main()
 except Exception:
  if OUT.exists():(OUT/f'failure_{int(time.time())}.txt').write_text(traceback.format_exc());status('failed')
  raise
