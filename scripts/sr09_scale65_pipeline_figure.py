"""Matched six-column image/contour views for human inspection, no new segmentation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.nice(19)
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));import sr09_scale65_biology as c
RUN=ROOT/'results/sr09/scale65_rebuild_2026-10-27';OUT=RUN/'pipeline_figure';OUT.mkdir(exist_ok=True);OLD=RUN/'paired/samples';SIM=RUN/'sim/samples';audit=pd.read_csv(RUN/'scale_audit.csv');records=[];allviews=[]
def polys(base,method,ids):
 marker=json.loads((base/'mosaic'/method/'cellvit_complete.json').read_text());raw=json.loads(Path(marker['cells_path']).read_text())['cells'];layout=pd.read_csv(base/'mosaic'/method/'layout.csv');ncol=int(np.ceil(np.sqrt(len(layout))));result={pid:[] for pid in ids}
 for cell in raw:
  x,y=cell['centroid'];col=int((x-128)//640);row=int((y-128)//640);k=row*ncol+col;local=np.array([x-128-col*640,y-128-row*640])
  if k<0 or k>=len(layout) or (local<0).any() or (local>=512).any():continue
  pid=layout.patch_id.iloc[k]
  if pid in result:result[pid].append(np.array(cell['contour'])-[128+col*640,128+row*640])
 return result
for sid in ['GSM6505133','GSM6505134','GSM6505135']:
 base=OLD/sid;manifest=pd.read_csv(base/'patches.csv');ref=pd.read_csv(base/'spot_features_native.csv');eligible=ref[ref.n>=5];first=eligible.iloc[(eligible.area_median-eligible.area_median.median()).abs().argmin()];rest=eligible[eligible.patch_id!=first.patch_id];second=rest.iloc[(rest.n-rest.n.quantile(.9)).abs().argmin()];ids=[first.patch_id,second.patch_id];r=audit.set_index('gsm').loc[sid];off,stretch=np.zeros((len(manifest),2)),np.ones(2);native=polys(base,'native',ids);rs=polys(base,'sr',ids);rb=polys(base,'bicubic',ids);ss=polys(SIM/sid,'sr',ids);sb=polys(SIM/sid,'bicubic',ids);fig,axs=plt.subplots(2,6,figsize=(18,7),layout='constrained')
 for j,pid in enumerate(ids):
  ix=manifest.index[manifest.patch_id==pid][0];delta=off[ix];M=np.array([[stretch[0],0,delta[0]+255.5*(1-stretch[0])],[0,stretch[1],delta[1]+255.5*(1-stretch[1])]],np.float32)
  def real(im,co):return cv2.warpAffine(im,M,(512,512),borderValue=(255,255,255)),[(a-255.5)*stretch+255.5+delta for a in co]
  def image(b,m,lr=False):
   im=cv2.imread(str(b/'patches'/m/f'{pid}.png'))[...,::-1];return cv2.resize(im,(512,512),interpolation=cv2.INTER_NEAREST) if lr else im
  views=[(image(base,'native'),native[pid]),real(image(base,'lr',True),rb[pid]),(image(SIM/sid,'lr',True),sb[pid]),real(image(base,'sr'),rs[pid]),(image(SIM/sid,'sr'),ss[pid]),real(image(base,'bicubic'),rb[pid])];titles=['Native / native contours','True hires / projected BC contours','Simulated4um / projected BC contours','SR true / SR contours','SR simulated / SR contours','Bicubic true / BC contours']
  area=ref.set_index('patch_id').loc[pid].area_median;record=dict(gsm=sid,patch_id=pid,barcode=manifest.barcode.iloc[ix],selection='median area' if j==0 else 'near90thpercentile nuclear count',native_n=int(ref.set_index('patch_id').loc[pid].n),native_median_area_um2=float(area));records.append(record)
  for ax,(im,co),title in zip(axs[j],views,titles):
   ax.imshow(im)
   for a in co:ax.plot(np.r_[a[:,0],a[0,0]],np.r_[a[:,1],a[0,1]],color='#00ffff',lw=.45)
   ax.add_patch(plt.Circle((255.5,255.5),110,fill=False,color='yellow',lw=.8));ax.set_title(title,fontsize=8);ax.axis('off')
  axs[j,0].text(0,-.08,f'{sid} {pid}; median={area:.1f} um2',transform=axs[j,0].transAxes,fontsize=8);allviews.append((record,views,titles))
 fig.suptitle('Same128um field; yellow55um spot; cyan CellViT contours. LR contours projected, not independent LR segmentation.',fontsize=9)
 for ext in ['png','pdf']:fig.savefig(OUT/f'{sid}_pipeline.{ext}',dpi=200)
 plt.close(fig)
record,views,_=allviews[0];fig,ax=plt.subplots(figsize=(12,12),layout='constrained');ax.imshow(views[0][0]);zoom=[]
for i,a in enumerate(views[0][1]):
 area=cv2.contourArea(a.astype(np.float32))*.25**2;center=a.mean(0);ax.plot(np.r_[a[:,0],a[0,0]],np.r_[a[:,1],a[0,1]],color='cyan',lw=.8);ax.text(*center,f'{i+1}:{area:.1f}',color='yellow',fontsize=6);zoom.append(dict(nucleus_id=i+1,area_um2=area,x=center[0],y=center[1]))
ax.axis('off');ax.set_title(f"Native128um field: {record['gsm']} {record['patch_id']}\nLabels: nucleus ID : area um2; inspect fragmentation/debris, labels are not pathology verdicts")
for ext in ['png','pdf']:fig.savefig(OUT/f'native_contour_zoom.{ext}',dpi=220)
plt.close(fig);pd.DataFrame(zoom).to_csv(OUT/'native_zoom_nuclei.csv',index=False);pd.DataFrame(records).to_csv(OUT/'selected_spots.csv',index=False);(OUT/'README.md').write_text('Six128um views, two per donor, selected solely on native median area and near90thpercentile nucleus count. No SR performance selection. Yellow circle=55um spot. Real-input images and contours are mapped to native frame using documented pixel-center transform. Raw LR columns display nearest-neighbor enlarged pixels with contours projected from corresponding bicubic outputs; these are not independent LR CellViT results. Native128um zoom labels every nucleus area to assess fragmentation/debris visually; no automated pathology claim.\n')
print('FIGURES COMPLETE',flush=True)
