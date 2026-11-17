"""Audit existing candidate metadata and public file lists without data downloads."""
import os, re, json, hashlib, urllib.request, concurrent.futures
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
root=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]))
out=root/'results/sr09/A_audit'
if (out/'candidates.csv').exists():
    raise SystemExit('Existing audit preserved; use a new output version.')
src=root/'results/sr_survey/geo_visium_image_survey.csv'
s=pd.read_csv(src); s=s[s.mentions_visium & s.category.str.startswith('fullres') & (s.n_hires_png>0)].copy()
h=pd.read_csv(root/'results/sr_survey/hest_benchmark_candidates.csv').fillna('')
cal=pd.read_csv(root/'results/sr_survey/hires_calibration/geo_per_gse.csv').set_index('id')
local=Path('/Volumes/INTENSO/GEO_visium_hires')
def fetch(url):
    try:
        with urllib.request.urlopen(url,timeout=12) as response:
            return response.read(2000000).decode('utf-8','replace'), ''
    except Exception as exc:
        return '',type(exc).__name__+': '+str(exc)
def audit(item):
    g=item['gse'];base=f'https://ftp.ncbi.nlm.nih.gov/geo/series/{g[:-3]}nnn/{g}/suppl/'
    cached=root/'results/sr09/A_audit_broad_candidates_v0'/(g+'_filelist.txt')
    listing,error=(cached.read_text(), '') if cached.exists() else fetch(base+'filelist.txt')
    files=[]
    for line in listing.splitlines():
        cells=line.split('\t')
        if len(cells)>=4 and cells[0]=='File':
            try: size=int(cells[3])
            except ValueError: size=None
            files.append({'gse':g,'name':cells[1],'size_bytes':size,'source_listing':base+'filelist.txt'})
    names=[x['name'] for x in files]
    gsm=sorted(set(re.findall(r'GSM\d+',' '.join(names))))
    d=local/g
    existing=sorted(str(p) for p in d.iterdir() if p.is_file()) if d.is_dir() else []
    # Source links establish potential overlap, not patient independence.
    overlap=h[h.astype(str).apply(lambda col: col.str.contains(g,regex=False)).any(axis=1)]
    row=dict(item)
    row.update(listing_status='read' if listing else 'unavailable',listing_error=error,
        listed_files=len(files),listed_gsm_count=len(gsm),listed_gsm_ids='|'.join(gsm),
        expression_file_evidence='|'.join(n for n in names if re.search(r'matrix|\.h5(ad)?(\.gz)?$|\.rds',n,re.I)),
        coordinates_file_evidence='|'.join(n for n in names if re.search(r'positions|coordinates',n,re.I)),
        scalefactor_file_evidence='|'.join(n for n in names if 'scalefactor' in n.lower()),
        local_calibration_files='|'.join(existing),
        historical_hires_mpp=float(cal.loc[g,'mpp']) if g in cal.index else None,
        native_mpp_status='unverified',same_scan_pair_status='unverified_requires_dimensions_and_scalefactors',
        independent_patient_count='unknown',cancer_type_status='title_only_not_sample_verified',
        hest_overlap_ids='|'.join(overlap.id.astype(str)),
        tcga_patient_overlap_status='unknown_requires_patient_identifiers',
        B_eligibility='not_confirmed',C_eligibility='not_confirmed',
        data_downloaded=False)
    return row,files,listing
rows=[]; file_rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for row,files,listing in pool.map(audit,s.to_dict('records')):
        rows.append(row);file_rows.extend(files)
        (out/(row['gse']+'_filelist.txt')).write_text(listing)
        print(row['gse'],row['listing_status'],len(files),flush=True)
pd.DataFrame(rows).to_csv(out/'candidates.csv',index=False)
pd.DataFrame(file_rows,columns=['gse','name','size_bytes','source_listing']).to_csv(out/'file_inventory.csv',index=False)
needed=[dict(x,download_status='NOT_AUTHORIZED_NOT_DOWNLOADED') for x in file_rows if re.search(r'hires|scalefactor|positions|matrix|\.h5|\.tiff?|\.svs',x['name'],re.I)]
pd.DataFrame(needed).to_csv(out/'potential_downloads.csv',index=False)
config={'date_utc':datetime.now(timezone.utc).isoformat(),'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'n_candidates':len(rows),'file_lists_read':sum(x['listing_status']=='read' for x in rows),'no_data_downloads':True,'workers':2,'random_seed':None,'scientific_analysis':False,'plan':'docs/SR09_PLAN_DRAFT_v2.md','limitations':'GSE-level lists cannot establish per-GSM pairing, independent patients or physical mpp; absent filenames may be inside archives.'}
(out/'run_config.yaml').write_text(json.dumps(config,ensure_ascii=False,indent=2))
(out/'README.md').write_text(f'''# SR09-A inventory audit\n\n{len(rows)} candidate GSE tabulated; {config['file_lists_read']} GEO file lists readable. No images or expression matrices were downloaded and no SR inference was run.\n\nCandidates: candidates.csv; per-file names and sizes: file_inventory.csv; potentially needed files: potential_downloads.csv.\n''')
print(json.dumps(config,ensure_ascii=False))
