"""Read-only SR09-A image geometry and barcode audit; no model inference."""
import os,json,gzip,io,hashlib,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import tifffile,h5py
from PIL import Image
root=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]))
cfg=json.loads((root/'config/sr09_a_pairing.json').read_text())
out=root/cfg['output'];out.mkdir(parents=True,exist_ok=False)
source=Path(cfg['external_root'])
inv=pd.read_csv(root/'results/he/pdac_spatial_public_inventory.tsv',sep='\t').set_index('gsm')
hest=pd.read_csv(root/'results/sr_survey/hest_benchmark_candidates.csv').fillna('')
qc=pd.read_csv(root/'results/he/A9/geo_qc.tsv',sep='\t').set_index('gsm')
rows=[];evidence=[]
def small_read(p):
    raw=gzip.open(p,'rb').read() if p.suffix=='.gz' else p.read_bytes()
    evidence.append({'path':str(p),'sha256_uncompressed':hashlib.sha256(raw).hexdigest(),'bytes_read':len(raw)})
    return raw
for g,samples in cfg['samples'].items():
    for gsm in samples:
        directory=source/g/'samples'/gsm
        paths=[p for p in directory.iterdir() if p.is_file() and not p.name.startswith('._')]
        sfp=next(p for p in paths if 'scalefactors_json' in p.name)
        sf=json.loads(small_read(sfp));hp=next(p for p in paths if 'tissue_hires_image' in p.name)
        with gzip.open(hp,'rb') as f:
            with Image.open(f) as image: hw,hh=image.size
        tp=next(p for p in paths if p.name.lower().endswith(('.tif.gz','.tiff.gz')))
        with gzip.open(tp,'rb') as f: header=f.read(cfg['max_tiff_header_bytes'])
        tw=th=None;header_error=''
        try:
            with tifffile.TiffFile(io.BytesIO(header)) as tf:
                page=tf.pages[0];tw,th=int(page.imagewidth),int(page.imagelength)
        except Exception as exc: header_error=type(exc).__name__+': '+str(exc)
        pp=next(p for p in paths if 'tissue_positions' in p.name)
        raw=small_read(pp);pos=pd.read_csv(io.BytesIO(raw),header=None)
        if str(pos.iloc[0,0]).lower()=='barcode':pos=pos.iloc[1:].copy()
        pos.columns=['barcode','in_tissue','row','col','y','x'];pos['in_tissue']=pd.to_numeric(pos.in_tissue)
        pos[['x','y']]=pos[['x','y']].apply(pd.to_numeric)
        h5=[p for p in paths if 'filtered_feature_bc_matrix.h5' in p.name]
        if h5:
            with h5py.File(h5[0],'r') as f: barcodes={x.decode() if isinstance(x,bytes) else str(x) for x in f['matrix/barcodes'][:]}
            expression_path=str(h5[0]);expression_kind='filtered_h5_barcodes_only'
        else:
            bp=next(p for p in paths if 'barcodes.tsv' in p.name)
            barcodes=set(small_read(bp).decode().splitlines());expression_path=str(bp);expression_kind='mtx_barcode_sidecar'
        tissue=pos[pos.in_tissue==1];matched=tissue[tissue.barcode.isin(barcodes)]
        scale=sf['tissue_hires_scalef'];mpp=cfg['spot_diameter_um']/sf['spot_diameter_fullres']
        original_meta=inv.loc[gsm].to_dict() if gsm in inv.index else {}
        match=hest[hest.astype(str).apply(lambda col:col.str.contains(gsm,regex=False)|col.str.contains(g,regex=False)).any(axis=1)]
        row={'gse':g,'gsm':gsm,'title':original_meta.get('title',''),'patient_existing_group':original_meta.get('patient',''),'tissue':original_meta.get('tissue',''),'native_path':str(tp),'hires_path':str(hp),'scalefactor_path':str(sfp),'tiff_width':tw,'tiff_height':th,'tiff_header_error':header_error,'hires_width':hw,'hires_height':hh,'tissue_hires_scalef':scale,'width_residual_px':tw*scale-hw if tw else None,'height_residual_px':th*scale-hh if th else None,'native_mpp_from_55um_spot':mpp,'hires_mpp_from_55um_spot':mpp/scale,'mpp_source':'55um spot / spot_diameter_fullres; not independently verified scanner metadata','n_tissue_spots':len(tissue),'n_tissue_expression_barcodes_matched':len(matched),'expression_path':expression_path,'expression_kind':expression_kind,'coords_in_fullres_bounds':bool((matched.x.between(0,tw-1)&matched.y.between(0,th-1)).all()) if tw and th else None,'hest_explicit_gsm_gse_matches':'|'.join(match.id.astype(str)),'independence_status':'not established by absence of explicit accession match','pairing_status':'geometry_to_be_interpreted; pixel identity not tested','prior_A9_mpp':float(qc.loc[gsm,'um_per_px']) if gsm in qc.index else None}
        rows.append(row);pd.DataFrame(rows).to_csv(out/'paired_samples.csv',index=False)
        print(gsm,tw,th,hw,hh,'mpp',mpp,'barcodes',len(matched),'/',len(tissue),flush=True)
pd.DataFrame(evidence).to_csv(out/'small_input_hashes.csv',index=False)
(out/'run_config.json').write_text(json.dumps(dict(cfg,run_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),pandas=pd.__version__,tifffile=tifffile.__version__,scientific_inference=False),ensure_ascii=False,indent=2))
