"""Freeze local HEST inputs, scale audit, and expression program scores."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
from pathlib import Path
import sys,json,time,hashlib,platform
import numpy as np,pandas as pd,scanpy as sc,scipy.sparse as sp,tifffile
from scipy.spatial import cKDTree
ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]));sys.path.insert(0,str(ROOT/'scripts'))
from sr09_scale65_biology import PROGRAMS
OUT=ROOT/'results/sr09/hest_bounds_2026-10-29'
def gate():
 while os.getloadavg()[0]>8:time.sleep(10)
def main():
 inv=pd.read_csv(ROOT/'results/sr09/hires_reproduction_2026-10-29/local_hest_ffpe_visium_inventory.csv');inv=inv[inv.native_local&inv.expression_readable&(inv.technology=='Visium')];assert len(inv)==9
 cat=pd.read_csv('/Volumes/INTENSO_BIG/HEST1k/HEST_v1_3_0.csv').set_index('id');records=[]
 for r in inv.itertuples():
  gate();d=OUT/'inputs'/r.id;d.mkdir(parents=True,exist_ok=True);print('preflight',r.id,flush=True)
  a=sc.read_h5ad(r.expression_path);X=sp.csr_matrix(a.layers['counts'] if 'counts' in a.layers else a.X);source='layers/counts' if 'counts' in a.layers else 'X'
  assert X.data.min()>=0 and np.max(np.abs(X.data-np.rint(X.data)))<1e-5,(r.id,'X not raw integer counts')
  xy=np.asarray(a.obsm['spatial']);keep=np.asarray(a.obs['in_tissue']==1) if 'in_tissue' in a.obs else np.ones(a.n_obs,bool);a=a[keep].copy();X=X[keep];xy=xy[keep]
  pitch=float(np.median(cKDTree(xy).query(xy,k=2)[0][:,1]));mpp=100/pitch;meta=cat.loc[r.id];sfinfo={str(k):dict(v.get('scalefactors',{})) for k,v in a.uns.get('spatial',{}).items()};sf=None;sfkey=''
  for lib,v in sfinfo.items():
   if 'tissue_hires_scalef' in v:sf=float(v['tissue_hires_scalef']);sfkey=lib+'/tissue_hires_scalef';break
  hm=mpp/sf if sf else 4.25;sf=sf or mpp/hm
  paths=r.native_paths.split('|');native=paths[0]
  with tifffile.TiffFile(native) as tf:h,w=tf.pages[0].shape[:2]
  radius=27.5/mpp;center_ok=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h);covered=(xy[:,0]>=radius)&(xy[:,0]<w-radius)&(xy[:,1]>=radius)&(xy[:,1]<h-radius)
  pos=pd.DataFrame(dict(barcode=a.obs_names.astype(str),in_tissue=1,x=xy[:,0],y=xy[:,1]));pos['center_in_image']=center_ok;pos['image_eligible']=covered;pos['coverage_reason']=np.where(covered,'complete_55um_circle',np.where(center_ok,'circle_intersects_image_boundary','center_outside_image'));pos.to_csv(d/'positions.csv',index=False)
  genes=np.asarray(a.var_names.astype(str));total=np.asarray(X.sum(1)).ravel();Y=(sp.diags(1e6/np.maximum(total,1))@X).tocsr();Y.data=np.log1p(Y.data);scores=pd.DataFrame(dict(total_umi=total),index=a.obs_names);cov=[]
  for name,gs in PROGRAMS.items():
   ix=np.flatnonzero(np.isin(genes,gs.split()));scores[name]=np.asarray(Y[:,ix].mean(1)).ravel() if len(ix) else np.nan;cov.append(dict(program=name,n_present=len(set(genes[ix])),n_requested=len(gs.split()),genes=' '.join(sorted(set(genes[ix])))))
  scores.to_csv(d/'program_scores.csv');pd.DataFrame(cov).to_csv(d/'gene_coverage.csv',index=False)
  (d/'metadata.json').write_text(json.dumps(dict(spatial_scalefactors=sfinfo,counts_source=source,counts_integer_checked=True,expression_shape=list(X.shape),genes_first=list(genes[:10])),indent=2))
  records.append(dict(sample=r.id,organ=meta.organ,cancer=meta.oncotree_code,dataset_title=meta.dataset_title,disease_state=meta.disease_state,patient=meta.patient,native_path=native,expression_path=r.expression_path,native_width=w,native_height=h,native_mpp=mpp,catalogue_mpp=meta.pixel_size_um_estimated,neighbor_px=pitch,neighbor_um=100,hires_mpp=hm,hires_scalef=sf,hires_scale_source=sfkey or 'user_fallback_4.25um; no tissue_hires_scalef',n_spots=len(pos),n_centers_outside=int((~center_ok).sum()),n_image_eligible=int(covered.sum()),n_incomplete_circle=int((center_ok&~covered).sum()),counts_source=source,old_SR08_test=True))
  pd.DataFrame(records).to_csv(OUT/'sample_audit.csv',index=False);del a,X,Y
 (OUT/'preflight_complete.json').write_text(json.dumps(dict(samples=9,python=platform.python_version(),scanpy=sc.__version__),indent=2))
if __name__=='__main__':main()
