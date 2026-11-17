"""Prepare reproducible, symmetric, blinded ×8 existence annotations; keep the key local."""
from pathlib import Path
import sys
import json
import base64
import cv2
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from sr03b_eval_morphology import load_condition
from sr08e_annotation_prep import crop, tissue_ok
from hemorph.nucleus_match import mutual_nn
SEED = 20261024
OUT = ROOT / 'results/sr08/annotation_x8'

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    meta = pd.read_csv(ROOT / 'results/sr_survey/hest_benchmark_candidates.csv').set_index('id')
    rng = np.random.default_rng(SEED)
    pools, denominators, positives, backgrounds = [], [], [], []
    for method in ('bicubic', 'bicubic2'):
        ref = load_condition(method, 'native')
        rg = dict(tuple(ref.groupby('patch_id')))
        for label, cond in [('bicubic', 'real2_4.0_x8run01'), ('SwinIR', 'swinir8_real2_4.0_x8run01')]:
            pred = load_condition(method, cond)
            for pid, b in pred.groupby('patch_id'):
                a = rg.get(pid, ref.iloc[:0])
                sid = b['sample'].iloc[0]
                matched = set(mutual_nn(a[['x','y']].to_numpy(), b[['x','y']].to_numpy(),12).i_b) if len(a) else set()
                for j, (x,y) in enumerate(b[['x','y']].to_numpy()):
                    row = dict(sample=sid, patch_id=pid, x=x, y=y, method=label,
                               stratum='primary' if meta.loc[sid,'px'] <= .3 else 'generalization',
                               cat='positive' if j in matched else 'unmatched')
                    (positives if j in matched else pools).append(row)
                denominators.append(dict(sample=sid,method=label,n_pred=len(b),n_unmatched=len(b)-len(matched)))
            for pid, a in rg.items():
                sid = a['sample'].iloc[0]
                for _ in range(3):
                    x,y = rng.uniform(32,480,2)
                    if len(a) and np.linalg.norm(a[['x','y']].to_numpy()-[x,y],axis=1).min() >=32:
                        if tissue_ok(crop(sid,pid,x,y)):
                            backgrounds.append(dict(sample=sid,patch_id=pid,x=x,y=y,method=label,
                                stratum='control',cat='background'))
    pool = pd.DataFrame(pools)
    selected=[]
    allocations=[]
    for (method,stratum), group in pool.groupby(['method','stratum'],sort=True):
        counts=group['sample'].value_counts().sort_index()
        raw=100*counts/counts.sum()
        quota=np.floor(raw).astype(int)
        for sid in (raw-quota).sort_values(ascending=False,kind='stable').index[:100-int(quota.sum())]:
            quota[sid]+=1
        # Retain every sample so slice-level FDR can be estimated.
        for sid in quota[quota == 0].index:
            donors = quota[quota > 1].index
            donor = (quota.loc[donors] - raw.loc[donors]).idxmax()
            quota[donor] -= 1
            quota[sid] = 1
        assert int(quota.sum()) == 100 and (quota >= 1).all()
        for sid,n in quota.items():
            subset=group[group['sample']==sid]
            got=subset.iloc[rng.choice(len(subset),int(n),replace=False)]
            selected.extend(got.to_dict('records'))
            allocations.append(dict(sample=sid,method=method,stratum=stratum,n_pool=len(subset),n_sampled=int(n)))
    for candidates, n in [(positives,20),(backgrounds,20)]:
        selected.extend([candidates[i] for i in rng.choice(len(candidates),n,replace=False)])
    for i in rng.choice(len(selected),20,replace=False):
        selected.append(dict(selected[i],repeat_source=int(i)))
    order=rng.permutation(len(selected))
    items,key=[],[]
    source_ids={int(i):f'x8q{k+1:03d}' for k,i in enumerate(order)}
    for k,i in enumerate(order):
        row=selected[i]
        iid=f'x8q{k+1:03d}'
        ok,buf=cv2.imencode('.png',crop(row['sample'],row['patch_id'],row['x'],row['y']))
        if not ok: raise ValueError(iid)
        items.append(dict(id=iid,img='data:image/png;base64,'+base64.b64encode(buf).decode()))
        key.append(dict(id=iid,**row,source_id=source_ids.get(row.get('repeat_source'),'')))
    OUT.mkdir()
    (OUT/'items.json').write_text(json.dumps(items))
    pd.DataFrame(key).to_csv(OUT/'key.csv',index=False)
    pd.DataFrame(allocations).to_csv(OUT/'sampling_allocation.csv',index=False)
    den=pd.DataFrame(denominators).groupby(['sample','method'],as_index=False)[['n_pred','n_unmatched']].sum()
    den['automatic_FDR']=den.n_unmatched/den.n_pred
    den.to_csv(OUT/'denominators.csv',index=False)
    template=(ROOT/'scripts/sr08w_x8_annotation_template.html').read_text()
    (OUT/'index.html').write_text(template.replace('__ITEMS__',json.dumps(items)))
    (OUT/'README.md').write_text('''# ×8 blinded existence annotation

Seed: 20261024. Frozen protocol: results/sr08/PLAN_x8.md §3.
400 unmatched items: 100 per method per resolution stratum, proportional largest-remainder allocation by sample with a minimum of one item per sample; any zero quota receives one transferred from the largest overallocated donor (total unchanged). All predicted centers are eligible, including patch edges; padded white regions are outside the available native patch. 20 positive and 20 background controls plus 20 repeat items. Background controls are automatic-reference-negative controls, not established non-nuclei.
Only native H&E is displayed, encoded losslessly. The center cross marks the candidate. No SR images or method identities appear in the page. key.csv and sampling tables must stay local and must not be shared with raters.
Open index.html directly in a browser. Labels are saved in localStorage; export JSON after completion and keep exports in raw/. Different raters must use separate browser profiles or clear storage after exporting. Uncertain items require independent second-rater review. No final FDR verdict before review and sample bootstrap with conservative sensitivity analysis.
''')
    (OUT/'raw').mkdir()
    print(f'{OUT}: {len(items)} items',flush=True)
if __name__=='__main__':
    main()
