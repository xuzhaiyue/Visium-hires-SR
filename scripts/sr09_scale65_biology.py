"""End-to-end exploratory morphology-expression analysis; reuse segmentation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
import sys,json,gzip,io,time,hashlib,argparse
from pathlib import Path
import cv2,numpy as np,pandas as pd,scipy.sparse as sp,scipy.io
from scipy.stats import spearmanr,rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from hemorph.he_image import hed
OUT=ROOT/'results/sr09/scale65_rebuild_2026-10-27/C';OUT.mkdir(exist_ok=True);cv2.setNumThreads(2)
PROGRAMS={'EMT':'VIM FN1 COL1A1 COL1A2 COL3A1 SPARC TAGLN ACTA2 SNAI2 ZEB1 TWIST1 CDH2 MMP2 POSTN THBS2 LUM TGFBI','hypoxia':'VEGFA CA9 SLC2A1 PGK1 LDHA BNIP3 NDRG1 ADM ENO1 P4HA1 EGLN3 ANKRD37','proliferation':'MKI67 TOP2A CCNB1 CDK1 BIRC5 PCNA MCM2 TYMS CENPF UBE2C','classical':'TFF1 TFF2 LGALS4 CTSE AGR2 CEACAM6 CLDN18','basal':'KRT5 KRT6A KRT17 S100A2 LY6D','interferon':'ISG15 IFI6 IFIT1 MX1 STAT1','immune':'PTPRC CD3E CD68 C1QA'}
FEATURES=['density','area_median','area_cv','eccentricity_median','h_mean','h_std','gray_lapvar']
def read(p):
 b=Path(p).read_bytes();return gzip.decompress(b) if str(p).endswith('.gz') else b
def expression(folder):
 folder=Path(folder);h5=list(folder.glob('*filtered_feature_bc_matrix.h5'))
 if h5:
  import scanpy as sc
  a=sc.read_10x_h5(h5[0]);X=sp.csr_matrix(a.X);genes=np.asarray(a.var_names);bcs=np.asarray(a.obs_names);paths=h5
 else:
  mt=next(folder.glob('*matrix.mtx*'));ft=next(folder.glob('*features.tsv*'));bc=next(folder.glob('*barcodes.tsv*'));paths=[mt,ft,bc]
  X=sp.csr_matrix(scipy.io.mmread(io.BytesIO(read(mt)))).T.tocsr();f=pd.read_csv(io.BytesIO(read(ft)),sep='\t',header=None);genes=f.iloc[:,1 if f.shape[1]>1 else 0].astype(str).to_numpy();bcs=pd.read_csv(io.BytesIO(read(bc)),sep='\t',header=None).iloc[:,0].astype(str).to_numpy()
 total=np.asarray(X.sum(1)).ravel();Y=sp.diags(1e6/np.maximum(total,1))@X;Y=Y.tocsr();Y.data=np.log1p(Y.data)
 scores=pd.DataFrame({'total_umi':total},index=bcs);coverage=[]
 for name,gs in PROGRAMS.items():
  wanted=gs.split();idx=np.flatnonzero(np.isin(genes,wanted));present=sorted(set(genes[idx]));coverage.append(dict(program=name,n_present=len(present),n_requested=len(wanted),genes=' '.join(present)))
  scores[name]=np.asarray(Y[:,idx].mean(1)).ravel() if len(idx) else np.nan
 assert scores.index.is_unique
 return scores,coverage,[str(p) for p in paths]
def offsets(r,p,manifest):
 # Compose native->hires and hires->LR half-pixel maps; crops used scale-only centers.
 sf=float(r.tissue_hires_scalef);m=float(r.native_mpp_from_55um_spot);w,h=p['lr_wh']
 q=np.array([float(r.hires_width)/w/sf,float(r.hires_height)/h/sf]);orig=np.array([r.native_width,r.native_height],float);hi=np.array([r.hires_width,r.hires_height],float)
 centers=manifest[['fullres_x','fullres_y']].to_numpy();off=(centers*(hi/(orig*sf)-1)+q/2-.5)*m/.25;stretch=q*m/4
 return off,stretch

def features(sid,base,r=None):
 dest=OUT/'A_paired'/sid if r is not None else OUT/'B_hires'/sid;dest.mkdir(parents=True,exist_ok=True)
 manifest=pd.read_csv(base/'patches.csv');p=json.loads((base/'prepared.json').read_text());off,stretch=offsets(r,p,manifest) if r is not None else (np.zeros((len(manifest),2)),np.ones(2))
 pd.DataFrame(dict(patch_id=manifest.patch_id,dx_um=off[:,0]*.25,dy_um=off[:,1]*.25)).to_csv(dest/'coordinate_offsets.csv',index=False)
 methods=['native','bicubic','sr'] if r is not None else ['bicubic','sr']
 for method in methods:
  save=dest/f'features_{method}.csv'
  if save.exists():continue
  print(sid,method,'features',flush=True);marker=json.loads((base/'mosaic'/method/'cellvit_complete.json').read_text());cells=json.loads(Path(marker['cells_path']).read_text())['cells'];layout=pd.read_csv(base/'mosaic'/method/'layout.csv');ncol=int(np.ceil(np.sqrt(len(layout))));groups={}
  for cell in cells:
   x,y=cell['centroid'];col=int((x-128)//640);row=int((y-128)//640);idx=row*ncol+col;local=np.array([x-128-col*640,y-128-row*640]);
   if idx<0 or idx>=len(layout) or (local<0).any() or (local>=512).any():continue
   contour=np.array(cell['contour'],np.float32)-[128+col*640,128+row*640];groups.setdefault(layout.patch_id.iloc[idx],[]).append((local,contour))
  del cells;rows=[]
  for i,pr in enumerate(manifest.itertuples()):
   shift=off[i] if method!='native' else np.zeros(2);scale=stretch if method!='native' else np.ones(2);im=cv2.imread(str(base/'patches'/method/f'{pr.patch_id}.png'))[...,::-1];mask=np.zeros((512,512),np.uint8);areas=[];ecc=[]
   for centroid,contour in groups.get(pr.patch_id,[]):
    corrected=(centroid-255.5)*scale+255.5+shift
    if np.linalg.norm(corrected-255.5)>110:continue
    areas.append(cv2.contourArea(contour.astype(np.float32))*.25**2*np.prod(scale));cv2.fillPoly(mask,[np.rint(contour).astype(np.int32)],1)
    if len(contour)>=5:
     _,(d1,d2),_=cv2.fitEllipse((contour*scale).astype(np.float32));ecc.append(np.sqrt(max(0,1-(min(d1,d2)/max(d1,d2))**2)) if max(d1,d2)>0 else np.nan)
   yy,xx=np.ogrid[:512,:512];circle=((xx-255.5)*scale[0]+shift[0])**2+((yy-255.5)*scale[1]+shift[1])**2<=110**2;inside=mask.astype(bool)&circle;H=hed(im)[...,0];gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;lap=cv2.Laplacian(gray,cv2.CV_32F);n=len(areas)
   rows.append(dict(patch_id=pr.patch_id,barcode=pr.barcode,n=n,density=n/(np.pi*.0275**2),area_median=np.median(areas) if n>=3 else np.nan,area_cv=np.std(areas)/np.mean(areas) if n>=3 else np.nan,eccentricity_median=np.nanmedian(ecc) if len(ecc)>=3 else np.nan,h_mean=H[inside].mean() if inside.any() else np.nan,h_std=H[inside].std() if inside.any() else np.nan,gray_lapvar=lap[circle].var()))
   if i%200==0:print(i,'/',len(manifest),flush=True)
  pd.DataFrame(rows).to_csv(save,index=False)
 return dest,methods

def correlate(sid,base,dest,methods,expr_folder,group):
 scores,cov,paths=expression(expr_folder);pd.DataFrame(cov).to_csv(dest/'gene_coverage.csv',index=False);(dest/'expression_sources.json').write_text(json.dumps(paths));manifest=pd.read_csv(base/'patches.csv');coords=manifest.set_index('barcode');scores=scores.reindex(coords.index);scores.to_csv(dest/'program_scores.csv');rows=[]
 assert scores.total_umi.notna().all(),f'{sid}: missing barcode'
 for method in methods:
  f=pd.read_csv(dest/f'features_{method}.csv').set_index('barcode');z=f.join(scores);z.to_csv(dest/f'spot_expression_{method}.csv')
  for feat in FEATURES:
   for prog in PROGRAMS:
    v=z[[feat,prog,'total_umi']+([] if feat=='density' else ['density'])].dropna();n=len(v)
    rho=spearmanr(v[feat],v[prog]).statistic if n>=10 else np.nan
    # Rank residualization gives partial Spearman; density itself only adjusts total UMI.
    cols=['total_umi']+([] if feat=='density' else ['density']);C=np.column_stack([np.ones(n)]+[rankdata(v[c]) for c in cols]);a=rankdata(v[feat]);b=rankdata(v[prog]);a=a-C@np.linalg.lstsq(C,a,rcond=None)[0];b=b-C@np.linalg.lstsq(C,b,rcond=None)[0];partial=np.corrcoef(a,b)[0,1] if n>=10 and a.std()>1e-8 and b.std()>1e-8 else np.nan
    rows.append(dict(group=group,gsm=sid,method=method,feature=feat,program=prog,n=n,rho=rho,partial_rho=partial,adjustment='total_UMI' if feat=='density' else 'total_UMI+nuclear_density'))
 pd.DataFrame(rows).to_csv(dest/'associations.csv',index=False)

def summary(group):
 folders=list((OUT/group).glob('*/associations.csv'));t=pd.concat([pd.read_csv(p) for p in folders],ignore_index=True);t.to_csv(OUT/f'{group}_associations.csv',index=False)
 agg=t.groupby(['method','feature','program']).agg(mean_rho=('rho','mean'),mean_partial=('partial_rho','mean'),n_samples=('gsm','nunique'),n_positive=('partial_rho',lambda x:int((x>0).sum())),n_negative=('partial_rho',lambda x:int((x<0).sum()))).reset_index();agg.to_csv(OUT/f'{group}_summary.csv',index=False)
 fig,axs=plt.subplots(1,len(t.method.unique()),figsize=(15,5),layout='constrained');axs=np.atleast_1d(axs)
 for ax,method in zip(axs,t.method.unique()):
  z=agg[agg.method==method].pivot(index='feature',columns='program',values='mean_partial').reindex(FEATURES);v=ax.imshow(z,vmin=-.5,vmax=.5,cmap='RdBu_r');ax.set_xticks(range(len(z.columns)),z.columns,rotation=60,ha='right');ax.set_yticks(range(len(z)),z.index);ax.set_title(method+' partial rho');
  for y in range(len(z)):
   for x in range(len(z.columns)):ax.text(x,y,f'{z.iloc[y,x]:.2f}',ha='center',va='center',fontsize=7)
 fig.colorbar(v,ax=list(axs),shrink=.65)
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{group}_heatmap.{ext}',dpi=160)
 plt.close(fig)
 # Rank candidate pairs by strength with consistent direction, without significance claims.
 sr=agg[agg.method=='sr'].copy();sr['consistency']=sr[['n_positive','n_negative']].max(axis=1)/sr.n_samples;sr['rank_score']=sr.mean_partial.abs()*sr.consistency;sr=sr.sort_values('rank_score',ascending=False);sr.to_csv(OUT/f'{group}_candidates.csv',index=False)
 for k,pair in enumerate(sr.head(5).itertuples()):
  candidates=t[(t.method=='sr')&(t.feature==pair.feature)&(t.program==pair.program)].copy();candidates['abs']=candidates.partial_rho.abs();best=candidates.sort_values('abs',ascending=False).iloc[0];sid=best.gsm;d=OUT/group/sid;z=pd.read_csv(d/'spot_expression_sr.csv').set_index('barcode');base=OUT.parent/'paired/samples'/sid if group=='A_paired' else OUT.parent/'hires/samples'/sid;coord=pd.read_csv(base/'patches.csv').set_index('barcode').reindex(z.index)
  cfg=json.loads((d/'image_sources.json').read_text());im=cv2.imdecode(np.frombuffer(read(cfg['hires']),np.uint8),cv2.IMREAD_COLOR)[...,::-1];sf=cfg['sf'];fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained');axs[0].imshow(im);axs[0].set_title(sid+' hires H&E');axs[0].axis('off')
  for ax,col in zip(axs[1:],[pair.feature,pair.program]):
   val=z[col];lo,hi=val.quantile([.02,.98]);v=ax.scatter(coord.fullres_x*sf,coord.fullres_y*sf,c=val,s=4,cmap='viridis',vmin=lo,vmax=hi);ax.set_xlim(0,im.shape[1]);ax.set_ylim(im.shape[0],0);ax.set_aspect('equal');ax.axis('off');ax.set_title(col);fig.colorbar(v,ax=ax,shrink=.6)
  fig.suptitle(f'Exploratory spatial co-variation; partial rho={best.partial_rho:.3f}')
  for ext in ['png','pdf']:fig.savefig(OUT/f'{group}_candidate{k+1}_{pair.feature}_{pair.program}.ma{ext}',format=ext,dpi=160)
  plt.close(fig)
 (OUT/f'{group}_README.md').write_text('# '+group+' morphology–expression associations\n\nAll associations (including weak ones) are in the associations table; heatmaps show sample-weighted mean partial correlations controlling total UMI and nuclear density (density features control UMI only). Programs: CPM 1e6, log1p, mean over each gene set; missing genes recorded per sample; no per-gene z-scoring. EMT contains stromal genes and is not tumour-cell EMT. The 7 paired samples are not 7 independent patients (GSE327056 patient count unknown). Maps are descriptive.\n\n'+sr[['feature','program','mean_rho','mean_partial','n_samples','n_positive','n_negative']].head(10).to_string(index=False)+'\n\nHaematoxylin mean/SD use the intersection of segmented nuclei and the 55 µm circle; grey-level Laplacian variance uses the whole spot and reflects stain and resolution as well as structure.')

def run_A():
 audit=pd.read_csv(ROOT/'results/sr09/A_pairing_2026-10-25/sample_decisions.csv')
 for r in audit.itertuples():
  base=OUT.parent/'paired/samples'/r.gsm;dest,methods=features(r.gsm,base,r);(dest/'image_sources.json').write_text(json.dumps(dict(hires=r.hires_path,sf=r.tissue_hires_scalef)));correlate(r.gsm,base,dest,methods,Path(r.expression_path).parent,'A_paired')
 summary('A_paired')
if __name__=='__main__':run_A()
