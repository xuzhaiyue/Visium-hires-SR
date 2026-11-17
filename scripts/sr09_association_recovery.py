"""Compare existing 7x7 Spearman association matrices with native references."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.nice(19)
from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd
from scipy.stats import pearsonr,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]));src=R/'results/sr09/scale65_rebuild_2026-10-27/C/A_paired_associations.csv';out=R/'results/sr09/association_recovery_2026-10-29';out.mkdir(exist_ok=True)
t=pd.read_csv(src);features=['density','area_median','area_cv','eccentricity_median','h_mean','h_std','gray_lapvar'];programs=['EMT','hypoxia','proliferation','classical','basal','interferon','immune'];keys=pd.MultiIndex.from_product([features,programs],names=['feature','program']);rows=[];elements=[]
assert not t.duplicated(['gsm','method','feature','program']).any()
for sid in sorted(t.gsm.unique()):
 tables={m:t[(t.gsm==sid)&(t.method==m)].set_index(['feature','program']).reindex(keys) for m in ['native','sr','bicubic']};mask=np.logical_and.reduce([np.isfinite(tables[m].rho.to_numpy()) for m in tables]);ref=tables['native'].rho.to_numpy();assert mask.sum()>2
 for m,v in tables.items():
  mat=v.rho.unstack('program').reindex(index=features,columns=programs);mat.to_csv(out/f'{sid}_{m}_spearman_matrix.csv')
  for (f,p),z,ok in zip(keys,v.rho,mask):elements.append(dict(gsm=sid,method=m,feature=f,program=p,spearman_rho=z,common_finite=bool(ok)))
 for m in ['sr','bicubic']:
  x=ref[mask];y=tables[m].rho.to_numpy()[mask];rows.append(dict(gsm=sid,method=m,n_matrix_elements_total=len(keys),n_common_finite=int(mask.sum()),matrix_element_pearson_r=pearsonr(x,y).statistic,matrix_element_spearman_r=spearmanr(x,y).statistic,mean_absolute_difference=float(np.mean(abs(x-y)))))
s=pd.DataFrame(rows);s.to_csv(out/'recovery_by_sample.csv',index=False);pd.DataFrame(elements).to_csv(out/'matrix_elements.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,col,title in zip(axs,['matrix_element_pearson_r','mean_absolute_difference'],['Association matrix agreement (higher is closer)','Mean absolute difference (lower is closer)']):
 z=s.pivot(index='gsm',columns='method',values=col)
 for sid,row in z.iterrows():ax.plot([0,1],[row.bicubic,row.sr],'-o',label=sid)
 ax.set_xticks([0,1],['Bicubic','SR']);ax.set_title(title,fontsize=10);ax.set_ylabel('Pearson r across matrix elements' if col.startswith('matrix') else 'Mean |method rho - native rho|')
axs[1].legend(fontsize=7,bbox_to_anchor=(1.02,1))
for ext in ['png','pdf','svg']:fig.savefig(out/f'association_recovery.{ext}',dpi=170)
summary=s.groupby('method').mean(numeric_only=True);summary.to_csv(out/'descriptive_mean.csv');p=s.pivot(index='gsm',columns='method');nr=int((p.matrix_element_pearson_r.sr>p.matrix_element_pearson_r.bicubic).sum());nd=int((p.mean_absolute_difference.sr<p.mean_absolute_difference.bicubic).sum())
config={'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input':str(src),'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'features':features,'programs':programs,'source_value':'rho = unadjusted spot Spearman; not partial_rho','agreement_primary':'Pearson correlation of corresponding Spearman matrix elements','agreement_additional':'Spearman correlation of matrix elements','difference':'mean absolute difference of matched rho elements','missing':'shared finite intersection across native/SR/bicubic per sample','thresholds':None,'no_new_inference':True};import yaml;(out/'run_config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False));(out/'README.md').write_text(f'''# Descriptive association recovery (2026-10-29)

Group A, 7 samples: per sample the 7 feature × 7 program Spearman matrices (49 elements) of native, SR and bicubic, taken directly from the precomputed unadjusted rho. SR/bicubic vs native use the common finite elements of all three methods, each element weighted equally.

Primary agreement = Pearson r between corresponding matrix elements; also element Spearman r; difference = mean absolute rho difference. Reported per sample; no thresholds, no significance tests, and the 49 elements are not treated as independent replicates.

SR matrix Pearson higher: {nr}/7; mean absolute difference lower: {nd}/7. Equal-weight means across samples:

{summary[['matrix_element_pearson_r','matrix_element_spearman_r','mean_absolute_difference']].round(4).to_string()}

Matrix correlation reflects the overall pattern; mean absolute difference also reflects magnitude. Global matrix agreement does not establish any single association. The 7 samples are not 7 independent patients.

recovery_by_sample.csv per-sample metrics; 21 matrix CSVs and matrix_elements.csv allow recomputation; association_recovery.png/pdf/svg. No training, inference or segmentation.
''');print(s.to_string(index=False));print(summary.to_string())
