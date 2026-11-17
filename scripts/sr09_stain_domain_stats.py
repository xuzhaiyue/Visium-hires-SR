"""Color/stain distributions in paired native/hires images and TCGA train tiles."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.nice(19)
import sys,json,datetime,gzip,time
from pathlib import Path
import cv2,numpy as np,pandas as pd
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));from hemorph.he_image import hed
OUT=ROOT/'results/sr09/B_sim_vs_real_2026-10-27';cv2.setNumThreads(2)
NAMES=['H_OD','E_OD','Lab_L','Lab_a','Lab_b'];EDGES=[np.linspace(0,3,301),np.linspace(0,3,301),np.linspace(0,100,301),np.linspace(-128,128,301),np.linspace(-128,128,301)]
class Stats:
 def __init__(self):self.n=0;self.fields=0;self.s=np.zeros(5);self.ss=np.zeros(5);self.hist=np.zeros((5,300),np.int64)
 def add(self,rgb,mask):
  v=rgb[mask].reshape(-1,3);hi=v.max(1).astype(float);lo=v.min(1).astype(float);v=v[((hi-lo)/np.maximum(hi,1)>.05)&(v.mean(1)<.95*255)]
  if len(v)>1024:v=v[np.linspace(0,len(v)-1,1024,dtype=int)]
  self.fields+=1
  if len(v)==0:return
  h=hed(v.reshape(-1,1,3)).reshape(-1,3);lab=cv2.cvtColor(v.astype(np.float32).reshape(-1,1,3)/255,cv2.COLOR_RGB2LAB).reshape(-1,3);z=np.column_stack([h[:,:2],lab]);self.n+=len(z);self.s+=z.sum(0);self.ss+=(z*z).sum(0)
  for k in range(5):self.hist[k]+=np.histogram(np.clip(z[:,k],EDGES[k][0],np.nextafter(EDGES[k][-1],-np.inf)),EDGES[k])[0]
 def rows(self,sample,source):
  result=[]
  for k,name in enumerate(NAMES):
   c=(EDGES[k][1:]+EDGES[k][:-1])/2;cdf=np.cumsum(self.hist[k])/max(self.n,1);q=np.interp([.05,.25,.5,.75,.95],cdf,c);result.append(dict(sample=sample,source=source,channel=name,n_pixels=self.n,n_fields=self.fields,mean=self.s[k]/self.n,std=np.sqrt(max(0,self.ss[k]/self.n-(self.s[k]/self.n)**2)),p05=q[0],p25=q[1],median=q[2],p75=q[3],p95=q[4]))
  return result

def gate():
 while os.getloadavg()[0]>8:time.sleep(10)
def rgb(path):return cv2.imread(str(path))[...,::-1]
config=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),native='all existing spot-centered native512 patches; central55um circle',hires='original hires PNG native pixels; same spot55um circles, no resize',TCGA='all5600 actual hr/train tiles only; center55um circle at.5mpp; no val/test',tissue_mask='RGB mean<242.25 and (max-min)/max>.05',pixel_sampling='deterministic evenly spaced up to1024 retained tissue pixels per field; pooled sampled pixel moments/histograms',HED='hemorph.he_image.hed; clipped log10 optical density Ruifrok matrix, same transform for all',LAB='OpenCV float RGB->CIELAB, L0-100 a/b signed',distance='mean channel Wasserstein distance/TCGA channel SD across H,E,L,a,b; descriptive similarity, not effect tuning',histograms='300 bins; H/E[0,3],L[0,100],a/b[-128,128]; moments unbinned; quantiles approximate from bins')
(OUT/'stain_run_config.json').write_text(json.dumps(config,indent=2)+'\n');stats={};manifest=[]
train=sorted((ROOT/'results/sr08/data/hr/train').glob('*.png'));assert len(train)==5600;t=Stats()
for i,f in enumerate(train):
 if i%100==0:gate();print('TCGA',i,len(train),flush=True)
 im=rgb(f);yy,xx=np.ogrid[:im.shape[0],:im.shape[1]];mask=(xx-(im.shape[1]-1)/2)**2+(yy-(im.shape[0]-1)/2)**2<=55**2;t.add(im,mask);manifest.append(dict(sample='TCGA_train',source='train',path=str(f)))
stats[('TCGA_train','train')]=t
audit=pd.read_csv(ROOT/'results/sr09/A_pairing_2026-10-25/sample_decisions.csv')
for r in audit.itertuples():
 print(r.gsm,flush=True);base=ROOT/'results/sr09/B_2026-10-25/samples'/r.gsm;p=pd.read_csv(base/'patches.csv');sn=Stats();sh=Stats();hires=cv2.imdecode(np.frombuffer(gzip.decompress(Path(r.hires_path).read_bytes()),np.uint8),cv2.IMREAD_COLOR)[...,::-1];sf=float(r.tissue_hires_scalef);mpp=float(r.native_mpp_from_55um_spot)/sf;rad=27.5/mpp
 for i,row in enumerate(p.itertuples()):
  if i%100==0:gate()
  f=base/'patches/native'/f'{row.patch_id}.png';im=rgb(f);yy,xx=np.ogrid[:512,:512];sn.add(im,(xx-255.5)**2+(yy-255.5)**2<=110**2)
  cx=(row.fullres_x+.5)*sf-.5;cy=(row.fullres_y+.5)*sf-.5;x0=max(0,int(np.floor(cx-rad)));x1=min(hires.shape[1],int(np.ceil(cx+rad))+1);y0=max(0,int(np.floor(cy-rad)));y1=min(hires.shape[0],int(np.ceil(cy+rad))+1);yy,xx=np.ogrid[y0:y1,x0:x1];sh.add(hires[y0:y1,x0:x1],(xx-cx)**2+(yy-cy)**2<=rad**2)
 stats[(r.gsm,'native')]=sn;stats[(r.gsm,'true_hires')]=sh;manifest.extend([dict(sample=r.gsm,source='native_spot_manifest',path=str(base/'patches.csv')),dict(sample=r.gsm,source='true_hires',path=r.hires_path)])
rows=[v for key,obj in stats.items() for v in obj.rows(*key)];pd.DataFrame(rows).to_csv(OUT/'stain_statistics.csv',index=False);pd.DataFrame(manifest).to_csv(OUT/'stain_input_manifest.csv',index=False);histrows=[];dist=[]
for key,obj in stats.items():
 dd=[]
 for k,name in enumerate(NAMES):
  centers=(EDGES[k][1:]+EDGES[k][:-1])/2;prob=obj.hist[k]/obj.n;ref=t.hist[k]/t.n;sd=np.sqrt(max(1e-12,t.ss[k]/t.n-(t.s[k]/t.n)**2));d=wasserstein_distance(centers,centers,prob,ref)/sd;dd.append(d)
  for j in range(300):histrows.append(dict(sample=key[0],source=key[1],channel=name,bin_center=centers[j],probability=prob[j]))
 dist.append(dict(sample=key[0],source=key[1],distance_to_TCGA=float(np.mean(dd)),**{name+'_distance':d for name,d in zip(NAMES,dd)}))
pd.DataFrame(histrows).to_csv(OUT/'stain_distributions.csv',index=False);ranking=pd.DataFrame(dist);ranking=ranking[ranking.source!='train'].sort_values('distance_to_TCGA');ranking['rank_within_source']=ranking.groupby('source').distance_to_TCGA.rank(method='min').astype(int);ranking.to_csv(OUT/'stain_TCGA_similarity.csv',index=False)
fig,axs=plt.subplots(1,5,figsize=(18,3.8),layout='constrained')
for k,ax in enumerate(axs):
 for key,obj in stats.items():
  c=(EDGES[k][1:]+EDGES[k][:-1])/2;ax.plot(c,obj.hist[k]/obj.n,label='/'.join(key),lw=2.5 if key[0]=='TCGA_train' else .65,alpha=1 if key[0]=='TCGA_train' else .65)
 ax.set_title(NAMES[k]);ax.set_xlim(np.interp(.005,np.cumsum(t.hist[k])/t.n,(EDGES[k][1:]+EDGES[k][:-1])/2),np.interp(.995,np.cumsum(t.hist[k])/t.n,(EDGES[k][1:]+EDGES[k][:-1])/2))
axs[-1].legend(fontsize=5)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'stain_distributions.{ext}',dpi=160)
plt.close(fig);print(ranking.to_string(index=False),flush=True);(OUT/'stain_complete.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),train_tiles=5600,paired_samples=7),indent=2))
