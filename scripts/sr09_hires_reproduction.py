"""Compare local native-to-hires resampling without model inference or downloads."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
from pathlib import Path
import io,gzip,json,time,hashlib,platform,traceback,inspect
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import numpy as np,pandas as pd,cv2,tifffile,skimage,scipy,PIL
from PIL import Image
from scipy.spatial import cKDTree
from skimage.transform import resize,rescale
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]))
OUT=ROOT/'results/sr09/hires_reproduction_2026-10-29'
cv2.setNumThreads(2)
def gate():
    while os.getloadavg()[0]>8:
        status('waiting_low_load',load=os.getloadavg()[0]);time.sleep(10)
def status(phase,**kw):
    obj=dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),phase=phase,**kw)
    (OUT/'status.json').write_text(json.dumps(obj,indent=2));print(obj,flush=True)
def read(p):
    return gzip.decompress(Path(p).read_bytes()) if str(p).endswith('.gz') else Path(p).read_bytes()
def png(p,im):
    cv2.imwrite(str(p),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
def quant(x):return np.rint(np.clip(x,0,255)).astype(np.uint8)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def masks(r,shape):
    raw=read(r.positions_path).decode();p=pd.read_csv(io.StringIO(raw),header=0 if raw.startswith('barcode') else None)
    p.columns=['barcode','in_tissue','row','col','y','x'];p=p[p.in_tissue==1]
    pitch=np.median(cKDTree(p[['x','y']]).query(p[['x','y']],k=2)[0][:,1])
    m=np.zeros(shape,np.uint8);sf=r.tissue_hires_scalef
    for row in p.itertuples():
        cv2.circle(m,(round((row.x+.5)*sf-.5),round((row.y+.5)*sf-.5)),round(pitch*sf/2),1,-1)
    inner=cv2.erode(m,np.ones((7,7),np.uint8)).astype(bool)
    centers=[];h,w=shape
    # Deterministic position-only views, requiring high tissue coverage where possible.
    candidates=[]
    for row in p.sort_values(['y','x']).itertuples():
        x=int(np.clip(round((row.x+.5)*sf-.5),64,w-64));y=int(np.clip(round((row.y+.5)*sf-.5),64,h-64))
        cov=float(m[y-64:y+64,x-64:x+64].mean())
        if cov>=.9:candidates.append((x,y,cov,row.barcode))
    if not candidates:
        for row in p.sort_values(['y','x']).itertuples():
            x=int(np.clip(round((row.x+.5)*sf-.5),64,w-64));y=int(np.clip(round((row.y+.5)*sf-.5),64,h-64))
            candidates.append((x,y,float(m[y-64:y+64,x-64:x+64].mean()),row.barcode))
    for q in (.25,.75):centers.append(candidates[round((len(candidates)-1)*q)])
    return m.astype(bool),inner,centers

def spectrum(im):
    gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float64)
    yy,xx=np.indices(gray.shape);rr=np.sqrt(np.fft.fftfreq(gray.shape[0])[:,None]**2+np.fft.fftfreq(gray.shape[1])[None,:]**2)
    z=(gray-gray.mean())*np.outer(np.hanning(gray.shape[0]),np.hanning(gray.shape[1]))
    ps=np.abs(np.fft.fft2(z))**2
    return np.array([ps[(rr>=lo)&(rr<hi)].mean() for lo,hi in ((.02,.1),(.1,.25),(.25,.5))])
def metrics(pred,real,mask,inner,views):
    d=pred.astype(np.float32)-real.astype(np.float32);v=d[mask];mse=float(np.mean(v*v,dtype=np.float64))
    a=cv2.Laplacian(cv2.cvtColor(pred,cv2.COLOR_RGB2GRAY).astype(np.float32),cv2.CV_32F,ksize=1)[inner]
    b=cv2.Laplacian(cv2.cvtColor(real,cv2.COLOR_RGB2GRAY).astype(np.float32),cv2.CV_32F,ksize=1)[inner]
    sp=[];sr=[]
    for x,y,*_ in views:sp.append(spectrum(pred[y-64:y+64,x-64:x+64]));sr.append(spectrum(real[y-64:y+64,x-64:x+64]))
    ratios=np.mean(sp,axis=0)/np.mean(sr,axis=0)
    shift,response=cv2.phaseCorrelate(cv2.cvtColor(real,cv2.COLOR_RGB2GRAY).astype(np.float32)*mask,cv2.cvtColor(pred,cv2.COLOR_RGB2GRAY).astype(np.float32)*mask)
    return dict(psnr_db=float(10*np.log10(255**2/mse)) if mse else float('inf'),mae=float(np.abs(v).mean()),max_abs=int(np.abs(v).max()),exact_rgb_pixel_fraction=float((v==0).all(axis=1).mean()),exact_channel_fraction=float((v==0).mean()),laplacian_variance_ratio=float(np.var(a)/np.var(b)),psd_low_ratio=ratios[0],psd_mid_ratio=ratios[1],psd_high_ratio=ratios[2],phase_shift_x=shift[0],phase_shift_y=shift[1],phase_response=response,n_tissue_pixels=int(mask.sum()))

def methods(native,shape):
    h,w=shape;nh,nw=native.shape[:2]
    for name,code in [('opencv_area',cv2.INTER_AREA),('opencv_linear',cv2.INTER_LINEAR),('opencv_cubic',cv2.INTER_CUBIC)]:
        gate();yield name,cv2.resize(native,(w,h),interpolation=code),'uint8 library output'
    pil=Image.fromarray(native)
    for name,code in [('pil_box',Image.Resampling.BOX),('pil_lanczos',Image.Resampling.LANCZOS),('pil_bicubic',Image.Resampling.BICUBIC)]:
        gate();yield name,np.asarray(pil.resize((w,h),resample=code)),'uint8 library output'
    del pil
    for order in (0,1,3):
        gate();out=np.empty((h,w,3),np.uint8)
        def plane(c):
            return quant(resize(native[...,c],(h,w),order=order,anti_aliasing=True,preserve_range=True,mode='reflect'))
        # Two single-thread scipy workers stay within the two-CPU budget.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for c,result in enumerate(pool.map(plane,range(3))):out[...,c]=result
        yield f'skimage_resize_rescale_aa_order{order}',out,'preserve_range; reflect; np.rint to uint8; rescale alias verified on fixture and output geometry'
    gate();out=np.empty((h,w,3),np.uint8);sx=(nw/w-1)/2;sy=(nh/h-1)/2
    for c in range(3):
        gate();blur=cv2.GaussianBlur(native[...,c].astype(np.float32),(0,0),sigmaX=sx,sigmaY=sy,borderType=cv2.BORDER_REFLECT_101)
        out[...,c]=quant(cv2.resize(blur,(w,h),interpolation=cv2.INTER_LINEAR));del blur
    yield 'gaussian_then_linear',out,'float32 Gaussian sigma=(native/target-1)/2; REFLECT_101; uint8 rint'

def draw(sid,method,real,pred,views):
    fig,ax=plt.subplots(2,4,figsize=(10.5,5.5),layout='constrained')
    for i,(x,y,cov,barcode) in enumerate(views):
        a=real[y-64:y+64,x-64:x+64];b=pred[y-64:y+64,x-64:x+64];d=np.abs(a.astype(float)-b.astype(float))
        ax[i,0].imshow(a);ax[i,1].imshow(b);im=ax[i,2].imshow(d.max(axis=2),vmin=0,vmax=20,cmap='magma')
        ax[i,3].imshow(np.clip((b.astype(float)-a.astype(float))*4+128,0,255).astype(np.uint8))
        for j,title in enumerate(['Real hires',method,'Max RGB absolute difference','Signed RGB difference ×4 +128']):ax[i,j].set_title(title,fontsize=8);ax[i,j].axis('off')
        ax[i,0].text(0,-.07,f'{barcode}; center=({x},{y}); tissue={cov:.2f}',transform=ax[i,0].transAxes,fontsize=6)
    fig.colorbar(im,ax=ax[:,2],label='RGB levels (clipped at 20)',shrink=.7)
    fig.suptitle(f'{sid}: two coordinate-selected 128×128 hires views',fontsize=11)
    for ext in ('png','pdf'):fig.savefig(OUT/f'{sid}_best_difference.{ext}',dpi=160)
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=pd.read_csv(ROOT/'results/sr09/scale65_rebuild_2026-10-27/scale_audit.csv');rows=rows[rows.paired==True]
    # rescale dispatches to resize after rounding the requested shape; verify locally.
    fixture=np.random.default_rng(2026).integers(0,256,(213,197,3),dtype=np.uint8);alias=[]
    for order in (0,1,3):
        a=resize(fixture,(19,18,3),order=order,preserve_range=True,anti_aliasing=True)
        b=rescale(fixture,(19/213,18/197),order=order,preserve_range=True,anti_aliasing=True,channel_axis=-1)
        alias.append(dict(order=order,max_difference=float(np.abs(a-b).max())))
    assert all(x['max_difference']==0 for x in alias)
    (OUT/'rescale_resize_alias.json').write_text(json.dumps(dict(fixture=alias,implementation_calls_resize='return resize(' in inspect.getsource(rescale)),indent=2))
    config=dict(run_date='2026-10-29',python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__,opencv=cv2.__version__,pillow=PIL.__version__,tifffile=tifffile.__version__,seed=2026,threads=2,nice=19,load_pause_above=8,source_audit=str(ROOT/'results/sr09/scale65_rebuild_2026-10-27/scale_audit.csv'),sources=rows.fillna('').to_dict('records'),mask='in_tissue center disk union, radius=median nearest neighbor pitch /2; erode3px for Laplacian',psd='two fixed 128x128 hires tissue windows; grayscale; remove mean; Hann2D; mean power in [0.02,.1),[.1,.25),[.25,.5) cycles/pixel; ratio of average predicted/reference powers',alignment='actual target shape; half-pixel center convention of resize; no fitted registration/color correction',rounding='float algorithms np.rint clipped 0..255; integer outputs unchanged',no_download=True,no_sr=True)
    (OUT/'run_config.json').write_text(json.dumps(config,indent=2))
    for r in rows.itertuples():
        sid=r.gsm;done=OUT/f'{sid}_complete.json'
        if done.exists():continue
        gate();status('decode_native',sample=sid)
        buf=io.BytesIO(read(r.native_path))
        with tifffile.TiffFile(buf) as tf:
            page=tf.pages[0];meta=dict(shape=list(page.shape),photometric=str(page.photometric),orientation=str(page.tags.get('Orientation')),software=str(page.tags.get('Software')),description=str(page.description)[:2000]);native=page.asarray(maxworkers=2)
        del buf
        if native.shape[0] in (3,4) and native.shape[-1] not in (3,4):native=np.moveaxis(native,0,-1)
        native=native[...,:3];assert native.dtype==np.uint8
        real=cv2.imdecode(np.frombuffer(read(r.hires_path),np.uint8),cv2.IMREAD_COLOR)[...,::-1]
        mask,inner,views=masks(r,real.shape[:2]);png(OUT/f'{sid}_tissue_mask.png',np.repeat((mask*255).astype(np.uint8)[...,None],3,axis=2))
        geometry=dict(sample=sid,native_shape=list(native.shape),hires_shape=list(real.shape),scale=r.tissue_hires_scalef,sf_round_shape=[round(native.shape[0]*r.tissue_hires_scalef),round(native.shape[1]*r.tissue_hires_scalef)],effective_scale_y=real.shape[0]/native.shape[0],effective_scale_x=real.shape[1]/native.shape[1],metadata=meta,views=[list(v) for v in views],native_compressed_sha256=sha(r.native_path),hires_compressed_sha256=sha(r.hires_path),sf_sha256=sha(r.scalefactor_path))
        (OUT/f'{sid}_geometry.json').write_text(json.dumps(geometry,indent=2))
        records=[];best=None;bestmae=float('inf')
        for name,pred,note in methods(native,real.shape[:2]):
            status('compare',sample=sid,method=name)
            m=metrics(pred,real,mask,inner,views);records.append(dict(sample=sid,gse=r.gse,method=name,**m,rounding_note=note))
            pd.DataFrame(records).to_csv(OUT/f'{sid}_methods.csv',index=False)
            if m['mae']<bestmae:bestmae=m['mae'];best=(name,pred.copy())
            time.sleep(.2)
        del native
        draw(sid,best[0],real,best[1],views);png(OUT/f'{sid}_best_hires.png',best[1])
        done.write_text(json.dumps(dict(best_method=best[0],best_mae=bestmae,methods=len(records)),indent=2))
    allrows=pd.concat([pd.read_csv(p) for p in OUT.glob('GSM*_methods.csv')],ignore_index=True)
    allrows['rank_psnr']=allrows.groupby('sample').psnr_db.rank(ascending=False,method='min')
    allrows.to_csv(OUT/'method_comparison.csv',index=False)
    summary=allrows.groupby('method').agg(mean_psnr=('psnr_db','mean'),min_psnr=('psnr_db','min'),mean_mae=('mae','mean'),worst_max_abs=('max_abs','max'),mean_laplacian_ratio=('laplacian_variance_ratio','mean'),mean_rank=('rank_psnr','mean')).sort_values('mean_rank')
    summary.to_csv(OUT/'method_ranking.csv');allrows.sort_values('psnr_db',ascending=False).groupby('sample').head(1).sort_values('sample').to_csv(OUT/'best_by_sample.csv',index=False)
    status('complete',samples=len(rows),conditions=len(allrows))
if __name__=='__main__':
    try:main()
    except Exception:
        (OUT/f'failure_{int(time.time())}.txt').write_text(traceback.format_exc());status('failed');raise
