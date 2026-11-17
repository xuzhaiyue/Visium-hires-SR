"""Summarize all locked HEST bracket conditions without acceptance thresholds."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
from pathlib import Path
import json,hashlib,datetime,time
import numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]));OUT=ROOT/'results/sr09/hest_bounds_2026-10-29'
FEATURES=['density','area_median','area_cv','eccentricity_median','h_mean','h_std','gray_lapvar'];PROGRAMS=['EMT','hypoxia','proliferation','classical','basal','interferon','immune'];CONDITIONS=['upper_area','lower_aa0'];METHODS=['bicubic','old_SR08','calibrated_SR'];CN={'COAD':'colon adenocarcinoma','COADREAD':'colorectal adenocarcinoma','LNET':'lung neuroendocrine tumour','LUSC':'lung squamous cell carcinoma','HGSOC':'high-grade serous ovarian carcinoma','PRAD':'prostate adenocarcinoma','IDC':'invasive ductal breast carcinoma'}
def bio(root,sid):return OUT/'biology'/root/'A_paired'/sid

def plot_lines(table,metrics,names,filename,audit):
 fig,axs=plt.subplots(2,len(metrics),figsize=(5*len(metrics)+2,7),layout='constrained',squeeze=False)
 colors=plt.get_cmap('tab10');ids=audit['sample'].tolist()
 for ri,condition in enumerate(CONDITIONS):
  for ci,(metric,title) in enumerate(zip(metrics,names)):
   ax=axs[ri,ci]
   for j,sid in enumerate(ids):
    z=table[(table['sample']==sid)&(table.condition==condition)]
    if 'feature' in z:z=z[z.feature==metric];values=z.set_index('method').rho.reindex(METHODS)
    else:values=z.set_index('method')[metric].reindex(METHODS)
    cancer=audit.set_index('sample').loc[sid,'cancer'];ax.plot(range(3),values,'o-',lw=1,ms=3,color=colors(j),label=f'{sid} ({cancer})')
   ax.set_xticks(range(3),METHODS,fontsize=8);ax.set_title(condition+' / '+title,fontsize=10);ax.set_ylabel('Spearman rho to native' if 'feature' in table else title);ax.spines[['top','right']].set_visible(False)
 axs[0,-1].legend(fontsize=7,bbox_to_anchor=(1.02,1),loc='upper left',frameon=False)
 for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{filename}.{ext}',dpi=170)
 plt.close(fig)

def main():
 audit=pd.read_csv(OUT/'sample_audit.csv');assert len(audit)==9
 checked=[]
 for sid in audit['sample']:
  units=[('reference','native')]+[(co,me) for co in CONDITIONS for me in ('bicubic','sr')]+[(co+'_old_SR08','sr') for co in CONDITIONS]
  for root,method in units:
   marker=OUT/root/'samples'/sid/'mosaic'/method/'cellvit_complete.json';z=json.loads(marker.read_text());path=Path(z['cells_path']);assert path.is_file() and path.stat().st_size>0
   h=hashlib.sha256()
   with path.open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''):
     while os.getloadavg()[0]>8:time.sleep(10)
     h.update(chunk)
   assert h.hexdigest()==z['cells_sha256'],str(path);checked.append(dict(sample=sid,root=root,method=method,cells_path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest(),matches_completion_hash=True))
 assert len(checked)==63;pd.DataFrame(checked).to_csv(OUT/'cellvit_output_verification.csv',index=False)
 metrics=[];recovery=[];assoc=[];elements=[];keys=pd.MultiIndex.from_product([FEATURES,PROGRAMS],names=['feature','program']);(OUT/'matrices').mkdir(exist_ok=True)
 for r in audit.itertuples(index=False):
  sid=r.sample;ref=pd.read_csv(bio('reference',sid)/'features_native.csv').set_index('barcode');tables={'native':ref};matrices={};native_assoc=None
  for condition in CONDITIONS:
   d=bio(condition,sid);a=pd.read_csv(d/'associations.csv');na=a[a.method=='native'].set_index(['feature','program']).rho.reindex(keys)
   if native_assoc is None:native_assoc=na;matrices['native']=na.to_numpy()
   else:assert np.allclose(native_assoc,na,equal_nan=True)
   for method,root,local in [('bicubic',condition,'bicubic'),('calibrated_SR',condition,'sr'),('old_SR08',condition+'_old_SR08','sr')]:
    name=condition+'__'+method;f=pd.read_csv(bio(root,sid)/f'features_{local}.csv').set_index('barcode').reindex(ref.index);assert f.n.notna().all();tables[name]=f
    aa=pd.read_csv(bio(root,sid)/'associations.csv');aa=aa[aa.method==local].copy();aa['method']=method;aa['condition']=condition;aa['sample']=sid;aa['cancer']=r.cancer;assoc.append(aa);matrices[name]=aa.set_index(['feature','program']).rho.reindex(keys).to_numpy()
    for feat in ['density','area_median','area_cv']:
     x=ref[feat].to_numpy();y=f[feat].to_numpy();eligible=np.ones(len(ref),bool) if feat=='density' else ref.n.to_numpy()>=5;valid=eligible&np.isfinite(x)&np.isfinite(y);bias=valid&(x!=0)
     rho=spearmanr(x[valid],y[valid]).statistic if valid.sum()>1 else np.nan
     metrics.append(dict(sample=sid,cancer=r.cancer,organ=r.organ,condition=condition,method=method,feature=feat,rho=rho,n_RNA_spots=r.n_spots,n_image_eligible=len(ref),n_reference_eligible=int(eligible.sum()),n_valid=int(valid.sum()),coverage=float(valid.sum()/eligible.sum()) if eligible.any() else np.nan,median_absolute_relative_error=float(np.median(np.abs(y[bias]/x[bias]-1))) if bias.any() else np.nan,n_zero_native_excluded_from_bias=int((valid&(x==0)).sum()),nominal_generated_hires_mpp=r.hires_mpp))
  common=np.logical_and.reduce([np.isfinite(v) for v in matrices.values()]);a=matrices['native'][common]
  for name,v in matrices.items():
   pd.Series(v,index=keys).unstack('program').reindex(index=FEATURES,columns=PROGRAMS).to_csv(OUT/'matrices'/f'{sid}__{name}.csv')
   for (feature,program),value,ok in zip(keys,v,common):elements.append(dict(sample=sid,cancer=r.cancer,method_condition=name,feature=feature,program=program,rho=value,common_finite=bool(ok)))
   if name=='native':continue
   condition,method=name.split('__');z=v[common];n=int(common.sum());r1=pearsonr(a,z).statistic if n>1 and np.std(a)>0 and np.std(z)>0 else np.nan;r2=spearmanr(a,z).statistic if n>1 else np.nan
   recovery.append(dict(sample=sid,cancer=r.cancer,organ=r.organ,condition=condition,method=method,n_matrix_elements_total=49,n_common_finite=n,matrix_element_pearson_r=r1,matrix_element_spearman_r=r2,mean_absolute_difference=float(np.mean(np.abs(a-z))) if n else np.nan))
 m=pd.DataFrame(metrics);rr=pd.DataFrame(recovery);assert len(m)==162 and len(rr)==54
 m.to_csv(OUT/'spot_metrics_by_sample.csv',index=False);rr.to_csv(OUT/'association_recovery_by_sample.csv',index=False);pd.DataFrame(elements).to_csv(OUT/'matrix_elements.csv',index=False);pd.concat(assoc).to_csv(OUT/'all_associations.csv',index=False)
 m.groupby(['condition','method','feature']).agg(mean_rho=('rho','mean'),n_samples=('sample','nunique'),mean_coverage=('coverage','mean'),mean_bias=('median_absolute_relative_error','mean')).to_csv(OUT/'spot_metrics_descriptive_means.csv')
 means=rr.groupby(['condition','method']).agg(mean_matrix_r=('matrix_element_pearson_r','mean'),mean_matrix_MAD=('mean_absolute_difference','mean'),n_samples=('sample','nunique'));means.to_csv(OUT/'association_recovery_descriptive_means.csv')
 plot_lines(m,['density','area_median','area_cv'],['Nuclear density','Median nuclear area','Nuclear area CV'],'spot_concordance',audit)
 plot_lines(rr,['matrix_element_pearson_r','mean_absolute_difference'],['Association matrix Pearson r','Matrix mean absolute difference'],'association_recovery',audit)
 cfg=json.loads((OUT/'run_config.json').read_text());import yaml;(OUT/'run_config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
 sampletable=audit[['sample','cancer','organ','n_spots','n_image_eligible','n_centers_outside','n_incomplete_circle','hires_mpp']].copy();sampletable['cancer_description']=sampletable.cancer.map(CN);sampletable.to_csv(OUT/'sample_cancers_and_coverage.csv',index=False)
 text='''# HEST human FFPE slides with two generated hires degradations (2026-10-29)

9 slides; the sharper (box/AREA) and blurrier (anti-aliased/AA0) generated hires were each processed with the locked calibrated SR, SR08 and bicubic, plus 9 native references: 63 CellViT conditions. No new training, weight selection, sigma fitting or downloads. The two generators bracket real hires sharpness; they are not strict statistical bounds and generated images are not real hires.

## Slides, scale and coverage

'''+sampletable.to_string(index=False)+'''

Where no original tissue_hires_scalef was available, 4.25 µm/px was used. Native WSI mpp was calibrated from the 100 µm nearest-neighbour spot pitch, with the HEST catalogue mpp kept for comparison. See sample_audit.csv and inputs/*/metadata.json.

Some native images do not cover all RNA coordinates. All barcodes are kept in inputs/*/positions.csv with center_in_image, image_eligible and coverage_reason; all methods are evaluated on the common set of spots whose 55 µm circle is fully covered. No spots were removed based on model performance.

Hires is generated from the whole native image; each spot uses the same centre and a 128 µm field: native 512 px at 0.25 µm; generated hires cropped to 32 px at 4 µm with half-pixel mapping for the two ×8 models; SR output 256 px interpolated to 512 px at 0.25 µm; bicubic follows the same 32→256→512 path. Statistics circle 55 µm, centre 255.5; density on all spots; area/CV missing when a method has <3 nuclei; the area/CV set is fixed by >=5 native nuclei.

## Outputs

- spot_metrics_by_sample.csv: 162 rows; Spearman rho, bias, coverage and n per slide / cancer / generator / method for density, area and CV; spot_concordance.png/.pdf/.svg.
- association_recovery_by_sample.csv: 54 rows; 63 7×7 matrices under matrices/; matrix_elements.csv and all_associations.csv; association_recovery.png/.pdf/.svg.
- Per-spot morphology features, expression programs and UMI: biology/*/A_paired/*/spot_expression_*.csv. All RNA barcodes and gene coverage are kept in inputs/.

Associations use unadjusted spot Spearman; programs are log1p CPM1e6 means of the 7 gene sets. Each matrix cell uses the spots available to each method (missing morphology is not zero-filled); the common finite elements of the 7×7 matrices are used for Pearson r, element Spearman and mean absolute rho difference against native. The 49 elements are not treated as independent replicates; no thresholds or significance tests.

## Descriptive means

'''+means.to_string()+'''

These nine slides were not used for training, TCGA validation-based weight selection or GSE327056 sigma calibration of the new model, so they are an external test for it; they include earlier SR08 test slides and are not treated as 9 independent patients.

The native image is a CellViT algorithmic reference, not expert ground truth. Texture metrics do not equal chromatin fidelity. Conclusions hold for the two generated degradations measured.

Run with nice 19, 2 threads, small batches with pauses. Model/source hashes and all parameters are in run_config.yaml; model-data isolation evidence in model_isolation.json.
'''
 (OUT/'README.md').write_text(text);(OUT/'verification.json').write_text(json.dumps(dict(samples=9,CellViT_conditions=63,spot_metric_rows=len(m),recovery_rows=len(rr),matrices=63,native_associations_identical_across_conditions=True,thresholds=None),indent=2))
 selected=[OUT/'README.md',OUT/'spot_metrics_by_sample.csv',OUT/'association_recovery_by_sample.csv',OUT/'run_config.yaml',OUT/'sample_audit.csv',Path(__file__)];(OUT/'delivery_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in selected},indent=2))
 print(means.to_string())
if __name__=='__main__':main()
