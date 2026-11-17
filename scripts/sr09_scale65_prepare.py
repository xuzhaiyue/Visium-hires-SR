"""Rebuild all SR GEO physical coordinates and fields using65um spot/100um pitch."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.nice(19)
from pathlib import Path
import sys,json,gzip,io,time
import numpy as np,pandas as pd,cv2,tifffile
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));import sr09_scale65_common as b
OUT=ROOT/'results/sr09/scale65_rebuild_2026-10-27';cv2.setNumThreads(2)
def read(p):
 data=Path(p).read_bytes();return gzip.decompress(data) if str(p).endswith('.gz') else data
def positions(path):
 raw=read(path).decode();p=pd.read_csv(io.StringIO(raw),header=0 if raw.startswith('barcode') else None);p.columns=['barcode','in_tissue','array_row','array_col','y','x'];return p[p.in_tissue==1].reset_index(drop=True)
def gate():
 while os.getloadavg()[0]>8:time.sleep(10)
def main():
 audit=pd.read_csv(ROOT/'results/sr09/A_pairing_2026-10-25_legacy_scale55/sample_decisions.csv');records=[]
 for r in audit.itertuples():
  pospath=next(Path(r.native_path).parent.glob('*tissue_positions*.gz'));records.append(dict(gse=r.gse,gsm=r.gsm,native_path=r.native_path,hires_path=r.hires_path,scalefactor_path=r.scalefactor_path,positions_path=str(pospath),expression_folder=str(Path(r.expression_path).parent),native_width=r.native_width,native_height=r.native_height,paired=True))
 picks=pd.read_csv(ROOT/'results/sr_validation/picks.csv');picks=picks[picks.gse.isin(['GSE274665','GSE282302','GSE299970','GSE337859'])]
 for r in picks.itertuples():
  d=Path('/Volumes/INTENSO/GEO_visium_validation')/r.gse;records.append(dict(gse=r.gse,gsm=r.gsm,native_path='',hires_path=str(d/r.hires),scalefactor_path=str(d/r.scalefactors),positions_path=str(d/r.positions),expression_folder=str(d),paired=False))
 for r in records:
  sf=json.loads(read(r['scalefactor_path']));p=positions(r['positions_path']);m=65/float(sf['spot_diameter_fullres']);scale=float(sf['tissue_hires_scalef']);dist=cKDTree(p[['x','y']]).query(p[['x','y']],k=2)[0][:,1];pitch=np.median(dist[dist>0]);r.update(native_mpp=m,hires_mpp=m/scale,tissue_hires_scalef=scale,spot_diameter_fullres=sf['spot_diameter_fullres'],neighbor_pitch_fullres_px=float(pitch),neighbor_pitch_um=float(pitch*m),neighbor_implied_mpp=float(100/pitch),scale_relative_difference=float((100/pitch)/m-1),n_spots=len(p),feature_circle_um=55,calibration_only=r['gse']=='GSE327056',heldout_test=r['gse']=='GSE211895')
 pd.DataFrame(records).to_csv(OUT/'scale_audit.csv',index=False)
 for r in records:
  sid=r['gsm'];dst=OUT/('paired' if r['paired'] else 'hires')/'samples'/sid;dst.mkdir(parents=True,exist_ok=True)
  if (dst/'prepared.json').exists():continue
  p=positions(r['positions_path']);hires=cv2.imdecode(np.frombuffer(read(r['hires_path']),np.uint8),cv2.IMREAD_COLOR)[...,::-1];m=r['native_mpp'];sf=r['tissue_hires_scalef'];hm=r['hires_mpp'];native=None
  if r['paired']:
   temp=dst/'native_temporary.tif'
   if not temp.exists():
    with gzip.open(r['native_path'],'rb') as src,temp.open('wb') as target:
     while True:
      chunk=src.read(8*1024*1024)
      if not chunk:break
      gate();target.write(chunk)
   with tifffile.TiffFile(temp) as tf:native=tf.pages[0].asarray(maxworkers=2)
   if native.shape[0] in [3,4] and native.shape[-1] not in [3,4]:native=np.moveaxis(native,0,-1)
   native=native[...,:3]
  rows=[]
  for i,row in enumerate(p.itertuples()):
   if i%100==0:gate();print(sid,'prepare',i,len(p),flush=True)
   pid=f'p{i:05d}';lr=b.crop_at(hires,(row.x+.5)*sf-.5,(row.y+.5)*sf-.5,32,4/hm,4/hm);lp=dst/'patches/lr'/f'{pid}.png';bp=dst/'patches/bicubic'/f'{pid}.png'
   if not lp.exists():b.png(lp,lr)
   if not bp.exists():b.png(bp,cv2.resize(cv2.resize(lr,(256,256),interpolation=cv2.INTER_CUBIC),(512,512),interpolation=cv2.INTER_CUBIC))
   if native is not None:
    path=dst/'patches/native'/f'{pid}.png'
    if not path.exists():b.png(path,b.crop_at(native,row.x,row.y,512,.25/m,.25/m))
   rows.append(dict(patch_id=pid,barcode=row.barcode,fullres_x=row.x,fullres_y=row.y,field_um=128,lr_sha256=b.digest(lp)))
  pd.DataFrame(rows).to_csv(dst/'patches.csv',index=False);(dst/'prepared.json').write_text(json.dumps(dict(n_patches=len(p),native_mpp=m,hires_mpp=hm,hires_scalef=sf,feature_circle_um=55,field_um=128,mapping='65um physical spot diameter; hires mpp=native_mpp/sf; correct half-pixel source mapping; direct cubic physical crop32@4um, native512@.25um'),indent=2))
  if native is not None:del native;temp.unlink()
 (OUT/'prepare_complete.json').write_text(json.dumps(dict(samples=len(records),paired=7,hires_only=4)))
if __name__=='__main__':main()
