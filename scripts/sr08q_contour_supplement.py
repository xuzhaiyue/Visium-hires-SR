"""Prepare approved ×4 supplemental native-only contour targets without replacing existing evidence."""
from pathlib import Path
import sys,json,base64,hashlib
from datetime import datetime,timezone
import cv2
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from sr08g_contour_prep import crop
OUT=ROOT/'results/sr08/contours_x4_supplement_2026-10-23'
SEED=20261023

def main():
    cv2.setNumThreads(2)
    OUT.mkdir(exist_ok=False)
    audit=ROOT/'results/sr08/contour_coverage_x4_2026-10-23'
    pop=pd.read_csv(audit/'eligible_population.csv')
    quotas=pd.read_csv(audit/'coverage_and_proposed_gaps.csv')
    old=pd.read_csv(ROOT/'results/sr08/contours/key.csv')
    coords=lambda d,x,y:list(zip(d['sample'],d.patch_id,d[x].round(5),d[y].round(5)))
    excluded=set(coords(old,'cx','cy'))
    pop['old_target']=np.array([x in excluded for x in coords(pop,'x','y')])
    pop[['sample','patch_id','x','y','old_target']].to_csv(OUT/'old_target_exclusion_mask.csv',index=False)
    pool=pop[~pop.old_target]
    mapped=pd.read_csv(audit/'drawn_contours_reclassified.csv')
    exposure=mapped['sample'].value_counts().to_dict()
    rng=np.random.default_rng(SEED); chosen=[]; reserves=[]
    for row in quotas.itertuples():
        n=int(row.proposed_additional_n)
        if n==0:continue
        candidates=pool[pool['cat'].eq(row.cat)&pool.area_tertile.eq(row.area_tertile)].copy()
        candidates['random_order']=rng.random(len(candidates))
        for _ in range(n):
            candidates['existing_drawn_plus_selected']=candidates['sample'].map(exposure).fillna(0)
            candidates=candidates.sort_values(['existing_drawn_plus_selected','random_order'],kind='stable')
            selected=candidates.iloc[0]
            chosen.append(selected.to_dict());sid=selected['sample'];exposure[sid]=exposure.get(sid,0)+1
            candidates=candidates.iloc[1:]
        reserves.extend(candidates.head(3).to_dict('records'))
    assert len(chosen)==29
    items=[];key=[]
    for k,i in enumerate(rng.permutation(len(chosen))):
        row=chosen[i]; iid=f'x4c{k+1:03d}'
        image,x0,y0=crop(row['sample'],row['patch_id'],row['x'],row['y'])
        assert image.shape==(128,128,3)
        ok,buf=cv2.imencode('.png',image)
        assert ok
        items.append(dict(id=iid,img='data:image/png;base64,'+base64.b64encode(buf).decode()))
        key.append(dict(id=iid,sample=row['sample'],patch_id=row['patch_id'],cx=row['x'],cy=row['y'],
            crop_x0=x0,crop_y0=y0,cat_x4=row['cat'],area_tertile=int(row['area_tertile']),ref_area_um2=row['ref_area_um2']))
    pd.DataFrame(key).to_csv(OUT/'key.csv',index=False)
    pd.DataFrame(reserves).to_csv(OUT/'reserve_candidates_private.csv',index=False)
    (OUT/'items.json').write_text(json.dumps(items))
    (OUT/'index.html').write_text((ROOT/'scripts/sr08q_contour_template.html').read_text().replace('__ITEMS__',json.dumps(items)))
    config=dict(run_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,n_items=29,crop_size_px=128,crop_mpp=.25,selection='fixed approved category/tertile quotas, prioritize least previously drawn sample then seeded random tie-break; exclude all prior 90 target coordinates',quotas=quotas.to_dict('records'),sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [audit/'eligible_population.csv',ROOT/'results/sr08/contours/key.csv']})
    (OUT/'run_config.yaml').write_text(json.dumps(config,indent=2))
    (OUT/'raw').mkdir()
    (OUT/'README.md').write_text('''# ×4 supplementary contours (29 items)

Quota: bicubic-only 15, both 3, missed 11; only the native image is shown (no method, predicted contour or category). Area tertiles within each category come from the coverage audit. Fixed seed 20261023; all coordinates of the earlier 90 items excluded; samples with fewer existing contours preferred. Private keys, reserves and masks are not exposed to the page.

Click points along the nuclear boundary or drag to draw, then save and go to the next item. The cross marks the target only; if the boundary is not visible, mark the item as untraceable rather than guessing. Output pts are 128 px crop coordinates at 0.25 µm/px. The 57 earlier contours are kept. Untraceable items are not deleted or replaced automatically.

Drafts and finished contours are stored in separate localStorage; the exported JSON contains all drafts and states. Drafts are not treated as traced contours.
''')
    print(pd.DataFrame(key).groupby(['cat_x4','area_tertile']).size().to_string())
if __name__=='__main__':main()
