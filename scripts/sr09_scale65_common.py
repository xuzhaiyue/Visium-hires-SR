"""Execute frozen SR09-B with serial, resumable low-load stages and private labels."""
import os,sys,json,time,gzip,io,hashlib,traceback,subprocess,argparse,base64
from pathlib import Path
from datetime import datetime,timezone
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import cv2,numpy as np,pandas as pd,tifffile
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
ROOT=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]))
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from hemorph import cellvit_io as CV,handcrafted as HC
from hemorph.nucleus_match import mutual_nn
C=json.loads((ROOT/'config/sr09_b.json').read_text());OUT=ROOT/C['output'];WORK=ROOT/'docs/worklog/2026-10-25_SR09_B'
cv2.setNumThreads(2);os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
D=pd.read_csv(ROOT/'results/sr09/A_pairing_2026-10-25_legacy_scale55/sample_decisions.csv');PRIMARY=D[D.gse=='GSE211895'].gsm.tolist();SAMPLES=PRIMARY+D[D.gse!='GSE211895'].gsm.tolist()
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def status(phase,**kw):
    x=dict(time_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),phase=phase,**kw)
    p=WORK/'status.tmp';p.write_text(json.dumps(x,indent=2));os.replace(p,WORK/'status.json');print(x,flush=True)
def gate():
    while os.getloadavg()[0]>C['load_threshold']:status('waiting_low_load',load=os.getloadavg()[0]);time.sleep(10)
def atomic_json(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));os.replace(tmp,p)
def png(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);assert cv2.imwrite(str(p),cv2.cvtColor(x,cv2.COLOR_RGB2BGR))
def rgb(p):return cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
def crop_at(im,cx,cy,n,sx,sy):
    # Output pixel centers map to a shared physical view centered at the spot.
    x=np.asarray(cx+(np.arange(n,dtype=np.float32)+.5-n/2)*sx,dtype=np.float32)
    y=np.asarray(cy+(np.arange(n,dtype=np.float32)+.5-n/2)*sy,dtype=np.float32)
    # OpenCV remap requires source dimensions below 32767; use the needed ROI.
    x0=max(0,int(np.floor(x.min()))-4);y0=max(0,int(np.floor(y.min()))-4)
    x1=min(im.shape[1],int(np.ceil(x.max()))+5);y1=min(im.shape[0],int(np.ceil(y.max()))+5)
    if x1<=x0 or y1<=y0:return np.full((n,n,3),255,np.uint8)
    return cv2.remap(im[y0:y1,x0:x1],*np.meshgrid(x-x0,y-y0),cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
def positions(row):
    p=next(Path(row.native_path).parent.glob('*tissue_positions*.gz'))
    with gzip.open(p,'rb') as f:b=f.read()
    a=pd.read_csv(io.BytesIO(b),header=None)
    if str(a.iloc[0,0]).lower()=='barcode':a=a.iloc[1:]
    a.columns=['barcode','in_tissue','row','col','y','x'];a[['in_tissue','y','x']]=a[['in_tissue','y','x']].apply(pd.to_numeric)
    return a[a.in_tissue==1].reset_index(drop=True)
def prepare(sid):
    dst=OUT/'samples'/sid
    if (dst/'prepared.json').exists():return
    status('preparing',sample=sid);gate();r=D.set_index('gsm').loc[sid];pos=positions(r)
    dst.mkdir(parents=True,exist_ok=True)
    with gzip.open(r.hires_path,'rb') as f:hires=cv2.imdecode(np.frombuffer(f.read(),np.uint8),cv2.IMREAD_COLOR)[...,::-1]
    mpp=float(r.native_mpp_from_55um_spot);sf=float(r.tissue_hires_scalef)
    w4=round(r.native_width*mpp/4);h4=round(r.native_height*mpp/4)
    low=cv2.resize(hires,(w4,h4),interpolation=cv2.INTER_AREA)
    ax=w4/r.native_width;ay=h4/r.native_height
    # Only transient decompression inside this run; source files remain read-only.
    temp=dst/'native_decode.tmp.tiff'
    if not (dst/'native_decode_complete.json').exists():
        with gzip.open(r.native_path,'rb') as f,temp.open('wb') as g:
            while True:
                b=f.read(8*1024*1024)
                if not b:break
                gate();g.write(b);time.sleep(.03)
        atomic_json(dst/'native_decode_complete.json',dict(source=r.native_path,bytes=temp.stat().st_size))
    with tifffile.TiffFile(temp) as tf:native=tf.pages[0].asarray(maxworkers=2)
    if native.ndim==3 and native.shape[0] in(3,4) and native.shape[-1] not in(3,4):native=np.moveaxis(native,0,-1)
    assert native.dtype==np.uint8 and native.shape[:2]==(int(r.native_height),int(r.native_width)) and native.shape[-1]>=3
    native=native[...,:3];manifest=[]
    for i,p in pos.iterrows():
        if i%50==0:gate();status('preparing_patches',sample=sid,done=i,total=len(pos))
        pid=f'p{i:05d}';lr=crop_at(low,p.x*ax,p.y*ay,32,1,1)
        ref=crop_at(native,p.x,p.y,512,.25/mpp,.25/mpp)
        lp=dst/'patches/lr'/f'{pid}.png';npth=dst/'patches/native'/f'{pid}.png'
        if not lp.exists():png(lp,lr)
        if not npth.exists():png(npth,ref)
        bp=dst/'patches/bicubic'/f'{pid}.png'
        if not bp.exists():png(bp,cv2.resize(cv2.resize(lr,(256,256),interpolation=cv2.INTER_CUBIC),(512,512),interpolation=cv2.INTER_CUBIC))
        manifest.append(dict(patch_id=pid,barcode=p.barcode,fullres_x=p.x,fullres_y=p.y,lr_sha256=digest(lp),native_png_sha256=digest(npth),field_um=128,center_model_x=256,center_model_y=256))
        if i%16==0:time.sleep(.05)
    pd.DataFrame(manifest).to_csv(dst/'patches.csv',index=False)
    qa=dst/'coordinate_qa';qa.mkdir(exist_ok=True)
    for j,i in enumerate(np.linspace(0,len(pos)-1,6,dtype=int)):
        p=pos.iloc[i];ref=rgb(dst/'patches/native'/f'p{i:05d}.png');hi=crop_at(hires,p.x*sf,p.y*sf,512,.25/(mpp/sf),.25/(mpp/sf))
        for im in (ref,hi):cv2.circle(im,(256,256),110,(0,255,255),1);cv2.drawMarker(im,(256,256),(0,255,255),cv2.MARKER_CROSS,16,1)
        canvas=np.concatenate([ref,hi],axis=1);png(qa/f'view_{j}.png',canvas)
    atomic_json(dst/'prepared.json',dict(n_patches=len(pos),native_shape=list(native.shape),native_mpp=mpp,hires_scalef=sf,lr_wh=[w4,h4],actual_lr_mpp_xy=[r.native_width*mpp/w4,r.native_height*mpp/h4],mapping='One 128um crop per in-tissue spot; original center -> shared physical view; 55um circle radius110 model pixels',source_native_path=r.native_path,source_hires_path=r.hires_path,source_hires_sha256=digest(r.hires_path),source_scalefactors_sha256=digest(r.scalefactor_path)))
    del native;temp.unlink();(dst/'native_decode_complete.json').unlink()
def infer(sid):
    import torch
    from sr08t_x8_core import build,tensor
    torch.set_num_threads(2);dst=OUT/'samples'/sid
    if (dst/'inference_complete.json').exists():return
    status('sr_inference',sample=sid);cfg=json.loads((ROOT/C['model_config']).read_text());model=build(cfg,False)
    model.load_state_dict(torch.load(ROOT/C['weights'],map_location='cpu',weights_only=True),strict=True);model.eval().to('mps')
    patches=pd.read_csv(dst/'patches.csv');todo=[p for p in patches.patch_id if not(dst/'patches/sr'/f'{p}.png').exists()]
    with torch.no_grad():
        for i in range(0,len(todo),C['batch']):
            gate();ids=todo[i:i+C['batch']];x=[rgb(dst/'patches/lr'/f'{p}.png') for p in ids]
            y=(model(tensor(x).to('mps')).clamp(0,1).cpu().numpy().transpose(0,2,3,1)*255+.5).astype(np.uint8)
            for p,im in zip(ids,y):png(dst/'patches/sr'/f'{p}.png',cv2.resize(im,(512,512),interpolation=cv2.INTER_CUBIC))
            if i%100==0:status('sr_inference',sample=sid,done=i+len(ids),total=len(todo))
            time.sleep(C['pause_seconds'])
    del model;torch.mps.empty_cache();atomic_json(dst/'inference_complete.json',dict(patches=len(patches),weights_sha256=C['weights_sha256']))
def mosaic(sid,method):
    dst=OUT/'samples'/sid/'mosaic'/method
    if (dst/'complete.json').exists():return
    status('mosaic',sample=sid,method=method);dst.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(OUT/'samples'/sid/'patches.csv');ncol=int(np.ceil(np.sqrt(len(p))));p['ox']=128+np.arange(len(p))%ncol*640;p['oy']=128+np.arange(len(p))//ncol*640
    im=np.full((128+int(np.ceil(len(p)/ncol))*640,128+ncol*640,3),255,np.uint8)
    for r in p.itertuples():im[r.oy:r.oy+512,r.ox:r.ox+512]=rgb(OUT/'samples'/sid/'patches'/method/f'{r.patch_id}.png')
    with tifffile.TiffWriter(dst/'he_sr09_0.25mpp.tiff',bigtiff=True) as writer:
        level=im;scale=1
        while True:
            gate();writer.write(level,tile=(512,512),photometric='rgb',compression='zlib',resolution=(40000/scale,40000/scale),resolutionunit='CENTIMETER',subfiletype=0 if scale==1 else 1,maxworkers=2)
            if min(level.shape[:2])//2<1024:break
            level=cv2.resize(level,(level.shape[1]//2,level.shape[0]//2),interpolation=cv2.INTER_AREA);scale*=2
    p.to_csv(dst/'layout.csv',index=False);atomic_json(dst/'complete.json',dict(n_patches=len(p),shape=list(im.shape)))
def segment(sid,method):
    dst=OUT/'samples'/sid/'mosaic'/method;cell=dst/'cellvit_sr09_0.25mpp'
    if (dst/'cellvit_complete.json').exists():
        assert len(list(cell.rglob('cells.json')))==1;return
    if cell.exists():cell.rename(dst/f'cellvit_interrupted_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}')
    status('cellvit',sample=sid,method=method);gate()
    env=dict(os.environ,MODEL='SAM',TAG='sr09_0.25mpp',BATCH='2',CPUS='2',RAY_WORKERS='1',RAY_CPUS='1',CELLVIT_TORCH_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',RAY_ENABLE_MAC_LARGE_OBJECT_STORE='1',RAY_GPUS_DEACTIVATION='1',PYTORCH_ENABLE_MPS_FALLBACK='1')
    args=[C['cellvit_python'],'scripts/sr09e_cellvit_lowload.py','--model','SAM','--outdir',str(cell),'--batch_size','2','--cpu_count','2','--ray_worker','1','--ray_remote_cpus','1','--memory','32000','--graph','--geojson','process_wsi','--wsi_path',str(dst/'he_sr09_0.25mpp.tiff'),'--wsi_mpp','0.25','--wsi_magnification','40']
    with (dst/'cellvit.log').open('a') as f:subprocess.run(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    assert len(list(cell.rglob('cells.json')))==1
    atomic_json(dst/'cellvit_complete.json',dict(cells_path=str(next(cell.rglob('cells.json'))),cells_sha256=digest(next(cell.rglob('cells.json')))))
def nuclei(sid,method):
    dst=OUT/'samples'/sid;file=dst/f'nuclei_{method}.csv'
    if file.exists():return pd.read_csv(file)
    cell,contours=CV.load_cells(dst/'mosaic'/method/'cellvit_sr09_0.25mpp');lay=pd.read_csv(dst/'mosaic'/method/'layout.csv');ncol=int(np.ceil(np.sqrt(len(lay))))
    col=((cell.cx-128)//640).astype(int).to_numpy();row=((cell.cy-128)//640).astype(int).to_numpy();idx=row*ncol+col
    x=cell.cx.to_numpy()-128-col*640;y=cell.cy.to_numpy()-128-row*640
    ok=(idx>=0)&(idx<len(lay))&(x>=0)&(x<512)&(y>=0)&(y<512)
    ids=np.flatnonzero(ok);shp=HC.shape_features([contours[i] for i in ids],.25)
    n=pd.DataFrame(dict(patch_id=lay.patch_id.to_numpy()[idx[ok]],x=x[ok],y=y[ok],area_um2=shp.area_um2.to_numpy()))
    n['within_spot']=((n.x-256)**2+(n.y-256)**2)<=110**2;n.to_csv(file,index=False);return n
FEATURES=['density','area_median','area_cv']
def evaluate(sid):
    dst=OUT/'samples'/sid
    if (dst/'metrics_complete.json').exists():return
    status('spot_features',sample=sid);p=pd.read_csv(dst/'patches.csv');tables={}
    for method in ['native','bicubic','sr']:
        n=nuclei(sid,method);g=n[n.within_spot].groupby('patch_id');rows=[]
        for r in p.itertuples():
            a=g.get_group(r.patch_id).area_um2.to_numpy() if r.patch_id in g.groups else np.array([]);k=len(a)
            rows.append(dict(patch_id=r.patch_id,barcode=r.barcode,n=k,density=k/(np.pi*.0275**2),area_median=float(np.median(a)) if k>=3 else np.nan,area_cv=float(np.std(a,ddof=C['cv_ddof'])/np.mean(a)) if k>=3 else np.nan))
        tables[method]=pd.DataFrame(rows).set_index('patch_id');tables[method].to_csv(dst/f'spot_features_{method}.csv')
    ref=tables['native'];metrics=[]
    for method in ['bicubic','sr']:
        for feat in FEATURES:
            eligible=np.ones(len(ref),bool) if feat=='density' else (ref.n>=5).to_numpy();a=ref[feat].to_numpy();b=tables[method][feat].to_numpy()
            valid=eligible&np.isfinite(a)&np.isfinite(b);bias=valid&(a!=0)
            metrics.append(dict(gsm=sid,group='primary' if sid in PRIMARY else 'reference',method=method,feature=feat,n_tissue=len(ref),n_reference_eligible=int(eligible.sum()),n_valid=int(valid.sum()),coverage=float(valid.sum()/eligible.sum()) if eligible.sum() else np.nan,rho=float(spearmanr(a[valid],b[valid]).statistic) if valid.sum()>=2 else np.nan,median_absolute_relative_error=float(np.median(np.abs(b[bias]/a[bias]-1))) if bias.any() else np.nan,n_zero_reference_excluded_from_bias=int((valid&(a==0)).sum())))
    pd.DataFrame(metrics).to_csv(dst/'metrics.csv',index=False);atomic_json(dst/'metrics_complete.json',dict(n_spots=len(ref)))
def summary():
    files=list((OUT/'samples').glob('*/metrics.csv'));metrics=pd.concat([pd.read_csv(p) for p in files]);metrics.to_csv(OUT/'metrics_by_sample.csv',index=False)
    rows=[]
    for feat in FEATURES:
        g=metrics[(metrics.group=='primary')&(metrics.feature==feat)];sr=g[g.method=='sr'].set_index('gsm').reindex(PRIMARY);bc=g[g.method=='bicubic'].set_index('gsm').reindex(PRIMARY)
        delta=sr.rho-bc.rho
        checks=dict(rho_at_least_two=bool((sr.rho>=.5).sum()>=2),rho_delta_mean=bool(delta.notna().all() and delta.mean()>=.10),rho_delta_at_least_two_positive=bool((delta>0).sum()>=2),bias_not_worse=bool(sr.median_absolute_relative_error.notna().all() and bc.median_absolute_relative_error.notna().all() and sr.median_absolute_relative_error.mean()<=bc.median_absolute_relative_error.mean()),coverage=bool(feat=='density' or (sr.coverage>=.5).sum()>=2))
        rows.append(dict(feature=feat,**checks,automatic_pass=all(checks.values()),human_check='pending',final_status='pending_human' if all(checks.values()) else 'failed_automatic',mean_delta_rho=float(delta.mean())))
    pd.DataFrame(rows).to_csv(OUT/'automatic_gates.csv',index=False)
    (OUT/'VERDICT.md').write_text('# SR09-B: automatic criteria\n\nPer-feature automatic criteria (pre-specified):\n\n'+pd.DataFrame(rows).to_string(index=False)+'\n\nFinal status awaits the 60-item expert check. The 4 coarser reference pairs are reported separately. No significance tests.\n')
    (OUT/'README.md').write_text('# SR09-B automatic evaluation\n\nProtocol: '+C['protocol']+'. The 3 primary donors and 4 coarser reference pairs are reported separately. Per-spot values, missingness, coverage and per-sample rho / median absolute relative error are in samples/ and metrics_by_sample.csv; per-criterion results in automatic_gates.csv / VERDICT.md.\n\nCV uses ddof=0; each spot uses a concentric 128 µm tile and nuclei are counted by centroid inside the 55 µm circle. Density uses all spots; area/CV use the fixed set with >=5 native nuclei, and a method with <3 nuclei is recorded as missing. Samples and weights were not selected by effect. Sources, versions/hashes and thread/load limits are in run_config.json.\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for ax,feat in zip(axes,FEATURES):
        g=metrics[(metrics.group=='primary')&(metrics.feature==feat)]
        for method in ['bicubic','sr']:
            z=g[g.method==method].set_index('gsm').reindex(PRIMARY)
            ax.plot(range(3),z.rho,'o-',label=method)
        ax.set_xticks(range(3),['DonorA','DonorB','DonorC']);ax.set_title(feat);ax.set_ylabel('Spearman rho');ax.axhline(.5,color='gray',lw=.6,ls='--');ax.legend(frameon=False)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'primary_concordance.{ext}',dpi=160)
    plt.close(fig)
def annotation():
    out=OUT/'annotation'
    if (out/'index.html').exists():return
    out.mkdir(exist_ok=True);rng=np.random.default_rng(C['seed']);selected=[];pools=[]
    for sid in PRIMARY:
        ref=nuclei(sid,'native');sr=nuclei(sid,'sr');a=ref.groupby('patch_id');candidates=[]
        for pid,b in sr[sr.within_spot].groupby('patch_id'):
            ar=a.get_group(pid) if pid in a.groups else ref.iloc[:0];matched=set(mutual_nn(ar[['x','y']].to_numpy(),b[['x','y']].to_numpy(),12).i_b) if len(ar) else set()
            for i,r in enumerate(b.itertuples()):
                if i not in matched:candidates.append(dict(gsm=sid,patch_id=pid,x=r.x,y=r.y))
        if len(candidates)<20:raise RuntimeError(f'{sid}: fewer than 20 eligible SR-only nuclei; no replacement sampling authorized')
        pools.append(dict(gsm=sid,n_pool=len(candidates),n_sampled=20))
        selected.extend(candidates[i] for i in rng.choice(len(candidates),20,replace=False))
    rng.shuffle(selected);items=[];key=[]
    for j,r in enumerate(selected):
        im=rgb(OUT/'samples'/r['gsm']/'patches/native'/f"{r['patch_id']}.png");crop=crop_at(im,r['x'],r['y'],256,1,1)
        ok,buf=cv2.imencode('.png',cv2.cvtColor(crop,cv2.COLOR_RGB2BGR));assert ok
        iid=f'bq{j+1:03d}';items.append(dict(id=iid,img='data:image/png;base64,'+base64.b64encode(buf).decode()));key.append(dict(id=iid,**r))
    pd.DataFrame(key).to_csv(out/'key.csv',index=False);pd.DataFrame(pools).to_csv(out/'sampling.csv',index=False)
    template=(ROOT/'scripts/sr08w_x8_annotation_template.html').read_text().replace('sr08_x8_labels_v1','sr09_b_20261025_labels_v1').replace('sr08_x8_v1','sr09_b_20261025_v1').replace('sr08_x8_labels_','sr09_b_labels_').replace('Blinded nucleus-presence labelling ×8','Blinded nucleus-presence labelling')
    (out/'index.html').write_text(template.replace('__ITEMS__',json.dumps(items)));(out/'raw').mkdir(exist_ok=True)
    atomic_json(out/'manifest.json',dict(seed=C['seed'],n_items=60,n_per_primary_donor=20,match_um=3,view_um=64,method_identity_hidden=True))
def run_all():
    assert (OUT/'pilot_coordinate_review.json').exists(),'Pilot coordinate images must be visually reviewed before further processing'
    for sid in SAMPLES:
        prepare(sid);infer(sid)
        for method in ['native','bicubic','sr']:mosaic(sid,method);segment(sid,method)
        evaluate(sid)
        if sid==PRIMARY[0]:atomic_json(OUT/'pilot_complete.json',dict(sample=sid,metrics=str(OUT/'samples'/sid/'metrics.csv')))
    summary();annotation()
    import socket
    with socket.socket() as sock:already_running=sock.connect_ex(('127.0.0.1',C['server_port']))==0
    if not already_running:
        with (WORK/'server.log').open('a') as log:
            server=subprocess.Popen([sys.executable,'scripts/sr09g_annotation_server.py'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(WORK/'server.json',dict(pid=server.pid,port=C['server_port']))
    status('automatic_complete_pending_page_qa_and_human_labels',url=f"http://127.0.0.1:{C['server_port']}/")
def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['pilot_prepare','all','summary']);args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(exist_ok=True)
    assert digest(ROOT/C['weights'])==C['weights_sha256']
    config=dict(C,protocol_sha256=digest(ROOT/C['protocol']),started_utc=datetime.now(timezone.utc).isoformat(),samples=SAMPLES,field_um=128,model_mpp=.25,spot_um=55,cv_ddof=0,source_code_sha256=digest(Path(__file__)))
    if (OUT/'run_config.json').exists():assert json.loads((OUT/'run_config.json').read_text())['protocol_sha256']==config['protocol_sha256']
    else:atomic_json(OUT/'run_config.json',config)
    if args.phase=='pilot_prepare':prepare(PRIMARY[0]);status('pilot_prepared_pending_visual_coordinate_review')
    elif args.phase=='all':run_all()
    else:summary();annotation()
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        WORK.mkdir(exist_ok=True);(WORK/f'exception_{int(time.time())}.txt').write_text(traceback.format_exc());status('FAILED',error=str(exc));raise
