"""SR09 cell-type-level analyses: reuse existing CellViT outputs (type field) only; no new segmentation, training or downloads.

1. spot cell-composition recovery (native / SR / bicubic) and type confusion
2. composition vs expression programs
3. tumour-nucleus (type=1) morphology vs expression programs
4. ridge-regression prediction with spatially blocked cross-validation
"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
try:
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
except PermissionError:
    pass
import sys, json, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr, rankdata
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / 'results/sr09'
OUT = R9 / 'celltype_application_2026-11-01'
OUT.mkdir(exist_ok=True)
cv2.setNumThreads(2)

TYPES = {1: 'neoplastic', 2: 'inflammatory', 3: 'connective', 4: 'dead', 5: 'epithelial'}
EPI_GENES = 'EPCAM KRT8 KRT18 KRT19 CDH1 CLDN4 CLDN7'.split()
RADIUS = 110  # 55 µm statistics-circle radius at 0.25 µm/px
CENTER = 255.5
MORPH = ['density', 'area_median', 'area_cv', 'eccentricity_median', 'h_mean', 'h_std', 'gray_lapvar']
PROGRAMS = ['EMT', 'hypoxia', 'proliferation', 'classical', 'basal', 'interferon', 'immune', 'epithelial']


def datasets():
    """(group, sample, condition, {method: mosaic_dir}, patches.csv, program_scores.csv, morph_feature_dir)"""
    rows = []
    s65 = R9 / 'scale65_rebuild_2026-10-27/paired/samples'
    for gsm in ['GSM6505133', 'GSM6505134', 'GSM6505135', 'GSM9647219', 'GSM9647220', 'GSM9647221', 'GSM9647222']:
        base = s65 / gsm
        feat = R9 / 'pitch_ablation_2026-10-28/C/A_paired' / gsm
        rows.append(dict(group='paired_real_hires', sample=gsm, condition='real_hires',
                         mosaics={m: base / 'mosaic' / m for m in ['native', 'bicubic', 'sr']},
                         patches=base / 'patches.csv', scores=feat / 'program_scores.csv', feat=feat,
                         expr=json.loads((feat / 'expression_sources.json').read_text())))
    hb = R9 / 'hest_bounds_2026-10-29'
    audit = pd.read_csv(hb / 'sample_audit.csv').set_index('sample')
    for cond in ['upper_area', 'lower_aa0']:
        for sid in audit.index:
            feat = hb / 'biology' / cond / 'A_paired' / sid
            rows.append(dict(group='HEST_generated_hires', sample=sid, condition=cond,
                             mosaics={'native': hb / 'reference/samples' / sid / 'mosaic/native',
                                      'bicubic': hb / cond / 'samples' / sid / 'mosaic/bicubic',
                                      'sr': hb / cond / 'samples' / sid / 'mosaic/sr'},
                             patches=hb / 'reference/samples' / sid / 'patches.csv', scores=feat / 'program_scores.csv',
                             feat=feat, expr=audit.loc[sid, 'expression_path'], cancer=audit.loc[sid].get('cancer', '')))
    return rows


def spot_cells(mosaic, manifest):
    """Assign nuclei to patches by mosaic layout; return nuclei inside the 55 µm circle (local coordinates, type, area, eccentricity, confidence)."""
    cells = json.loads(Path(json.loads((mosaic / 'cellvit_complete.json').read_text())['cells_path']).read_text())['cells']
    layout = pd.read_csv(mosaic / 'layout.csv')
    ncol = int(np.ceil(np.sqrt(len(layout))))
    pid = layout.patch_id.to_numpy()
    rec = []
    for c in cells:
        x, y = c['centroid']
        col = int((x - 128) // 640); row = int((y - 128) // 640); idx = row * ncol + col
        lx, ly = x - 128 - col * 640, y - 128 - row * 640
        if idx < 0 or idx >= len(layout) or not (0 <= lx < 512 and 0 <= ly < 512):
            continue
        if (lx - CENTER) ** 2 + (ly - CENTER) ** 2 > RADIUS ** 2:
            continue
        ct = np.asarray(c['contour'], np.float32)
        area = cv2.contourArea(ct) * 0.0625
        ecc = np.nan
        if len(ct) >= 5:
            _, (d1, d2), _ = cv2.fitEllipse(ct)
            if max(d1, d2) > 0:
                ecc = np.sqrt(max(0, 1 - (min(d1, d2) / max(d1, d2)) ** 2))
        rec.append((pid[idx], lx, ly, c['type'], area, ecc, c.get('type_prob', np.nan)))
    del cells
    df = pd.DataFrame(rec, columns=['patch_id', 'x', 'y', 'type', 'area', 'ecc', 'type_prob'])
    return df[df.patch_id.isin(manifest.patch_id)]


def spot_table(nuc, manifest):
    g = nuc.groupby('patch_id')
    t = pd.DataFrame(index=manifest.patch_id)
    t['n_total'] = g.size().reindex(t.index).fillna(0)
    for k, name in TYPES.items():
        t['n_' + name] = nuc[nuc.type == k].groupby('patch_id').size().reindex(t.index).fillna(0)
        t['p_' + name] = np.where(t.n_total >= 3, t['n_' + name] / t.n_total.clip(lower=1), np.nan)
    neo = nuc[nuc.type == 1].groupby('patch_id')
    n = neo.size().reindex(t.index).fillna(0)
    t['neo_area_median'] = neo.area.median().reindex(t.index).where(n >= 3)
    t['neo_area_cv'] = (neo.area.std(ddof=0) / neo.area.mean()).reindex(t.index).where(n >= 3)
    t['neo_ecc_median'] = neo.ecc.median().reindex(t.index).where(n >= 3)
    t['barcode'] = manifest.set_index('patch_id').barcode.reindex(t.index)
    return t.reset_index()


def confusion(nat, oth):
    """Type confusion after mutual-nearest-neighbour matching (3 µm = 12 px) of native and method nuclei within a spot."""
    out = np.zeros((6, 6), int)  # row/column 0 = unmatched
    ga = dict(tuple(nat.groupby('patch_id'))); gb = dict(tuple(oth.groupby('patch_id')))
    for p in set(ga) | set(gb):
        a = ga.get(p); b = gb.get(p)
        if a is None:
            for t_ in b.type: out[0, t_] += 1
            continue
        if b is None:
            for t_ in a.type: out[t_, 0] += 1
            continue
        A = a[['x', 'y']].to_numpy(); B = b[['x', 'y']].to_numpy()
        da, ia = cKDTree(B).query(A); db, ib = cKDTree(A).query(B)
        mut = (ib[ia] == np.arange(len(A))) & (da <= 12)
        ta = a.type.to_numpy(); tb = b.type.to_numpy()
        for i in np.flatnonzero(mut): out[ta[i], tb[ia[i]]] += 1
        for i in np.flatnonzero(~mut): out[ta[i], 0] += 1
        matched_b = set(ia[mut])
        for j in range(len(B)):
            if j not in matched_b: out[0, tb[j]] += 1
    return out


def epithelial_score(path, barcodes):
    if isinstance(path, list) and len(path) == 3:
        import gzip, io, scipy.io
        rd = lambda q: gzip.decompress(Path(q).read_bytes()) if str(q).endswith('.gz') else Path(q).read_bytes()
        mt, ft, bc = path
        X = sp.csr_matrix(scipy.io.mmread(io.BytesIO(rd(mt)))).T.tocsr()
        f = pd.read_csv(io.BytesIO(rd(ft)), sep='\t', header=None); genes = f.iloc[:, 1 if f.shape[1] > 1 else 0].astype(str).to_numpy()
        bcs = pd.read_csv(io.BytesIO(rd(bc)), sep='\t', header=None).iloc[:, 0].astype(str).to_numpy()
        path = None
    p = str(path[0] if isinstance(path, list) else path)
    if path is None:
        pass
    elif p.endswith('.h5ad'):
        import anndata
        a = anndata.read_h5ad(p)
        X = sp.csr_matrix(a.X); genes = np.asarray(a.var_names).astype(str); bcs = np.asarray(a.obs_names).astype(str)
    else:
        import scanpy as sc
        a = sc.read_10x_h5(p); X = sp.csr_matrix(a.X); genes = np.asarray(a.var_names); bcs = np.asarray(a.obs_names)
    tot = np.asarray(X.sum(1)).ravel()
    idx = np.flatnonzero(np.isin(genes, EPI_GENES))
    Y = (sp.diags(1e6 / np.maximum(tot, 1)) @ X[:, idx]).tocsr(); Y.data = np.log1p(Y.data)
    s = pd.Series(np.asarray(Y.mean(1)).ravel(), index=bcs)
    return s.reindex(barcodes).to_numpy(), int(len(idx))


def partial(x, y, covs):
    v = pd.concat([x, y] + covs, axis=1).dropna()
    if len(v) < 20: return np.nan, np.nan, len(v)
    rho = spearmanr(v.iloc[:, 0], v.iloc[:, 1]).statistic
    C = np.column_stack([np.ones(len(v))] + [rankdata(v.iloc[:, k]) for k in range(2, v.shape[1])])
    a = rankdata(v.iloc[:, 0]); b = rankdata(v.iloc[:, 1])
    a = a - C @ np.linalg.lstsq(C, a, rcond=None)[0]; b = b - C @ np.linalg.lstsq(C, b, rcond=None)[0]
    pr = np.corrcoef(a, b)[0, 1] if a.std() > 1e-8 and b.std() > 1e-8 else np.nan
    return rho, pr, len(v)


def cv_r2(X, y, blocks):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    pred = np.full(len(y), np.nan)
    for b in np.unique(blocks):
        tr = blocks != b; te = ~tr
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    t0 = time.time()
    comp_rows, conf_rows, expr_rows, neo_rows, pred_rows = [], [], [], [], []
    native_cache = {}
    for d in datasets():
        key = f"{d['sample']}_{d['condition']}"
        dest = OUT / 'per_sample' / key; dest.mkdir(parents=True, exist_ok=True)
        manifest = pd.read_csv(d['patches'])
        if 'image_eligible' in manifest: manifest = manifest[manifest.image_eligible.astype(bool)]
        scores = pd.read_csv(d['scores'], index_col=0)
        epi, n_epi = epithelial_score(d['expr'], scores.index.to_numpy()) if d['sample'] not in native_cache else native_cache[d['sample']]['epi']
        scores['epithelial'] = epi
        tabs, nucs = {}, {}
        for m, mos in d['mosaics'].items():
            f = dest / f'spot_types_{m}.csv'
            if m == 'native' and d['sample'] in native_cache:
                nucs[m] = native_cache[d['sample']]['nuc']; tabs[m] = native_cache[d['sample']]['tab']; tabs[m].to_csv(f, index=False); continue
            print(key, m, f'{time.time()-t0:.0f}s', flush=True)
            nucs[m] = spot_cells(mos, manifest); tabs[m] = spot_table(nucs[m], manifest); tabs[m].to_csv(f, index=False)
            time.sleep(2)
        native_cache[d['sample']] = dict(nuc=nucs['native'], tab=tabs['native'], epi=(epi, n_epi))
        meta = dict(group=d['group'], sample=d['sample'], condition=d['condition'], cancer=d.get('cancer', 'PDAC' if d['sample'].startswith('GSM') else ''))
        nat = tabs['native'].set_index('barcode')
        # 1. composition recovery
        for m in ['sr', 'bicubic']:
            z = tabs[m].set_index('barcode').reindex(nat.index)
            for name in TYPES.values():
                for kind in ['n_', 'p_']:
                    v = pd.concat([nat[kind + name], z[kind + name]], axis=1).dropna()
                    rho = spearmanr(v.iloc[:, 0], v.iloc[:, 1]).statistic if len(v) >= 20 and v.iloc[:, 0].std() > 0 else np.nan
                    comp_rows.append(dict(meta, method=m, cell_type=name, measure=kind.strip('_'), rho=rho, n=len(v),
                                          native_mean=nat[kind + name].mean(), method_mean=z[kind + name].mean()))
            cm = confusion(nucs['native'], nucs[m])
            for i in range(6):
                for j in range(6):
                    conf_rows.append(dict(meta, method=m, native_type=(['unmatched'] + list(TYPES.values()))[i],
                                          method_type=(['unmatched'] + list(TYPES.values()))[j], count=int(cm[i, j])))
        # 2/3. against expression
        sc = scores.reindex(nat.index)
        pairs = [('n_inflammatory', 'immune'), ('n_connective', 'EMT'), ('n_neoplastic', 'epithelial'),
                 ('n_neoplastic', 'proliferation'), ('p_inflammatory', 'immune'), ('p_connective', 'EMT'),
                 ('p_neoplastic', 'epithelial'), ('n_epithelial', 'epithelial')]
        for m in ['native', 'sr', 'bicubic']:
            z = tabs[m].set_index('barcode').reindex(nat.index)
            for feat, prog in pairs:
                rho, pr, n = partial(z[feat], sc[prog], [sc.total_umi])
                expr_rows.append(dict(meta, method=m, feature=feat, program=prog, rho=rho, partial_rho_umi=pr, n=n))
            for feat in ['neo_area_median', 'neo_area_cv', 'neo_ecc_median']:
                for prog in PROGRAMS:
                    rho, pr, n = partial(z[feat], sc[prog], [sc.total_umi, z.n_neoplastic])
                    neo_rows.append(dict(meta, method=m, feature=feat, program=prog, rho=rho,
                                         partial_rho_umi_neocount=pr, n=n))
        # 4. prediction: 7 morphology features + type fractions/counts; 5 spatial blocks (x-coordinate quintiles)
        mf = {m: pd.read_csv(d['feat'] / f'features_{m}.csv').set_index('barcode').reindex(nat.index) for m in ['native', 'sr', 'bicubic']}
        xy = manifest.set_index('barcode').reindex(nat.index)
        blocks = pd.qcut(xy.fullres_x.rank(method='first'), 5, labels=False).to_numpy()
        tcols = [c for c in nat.columns if c.startswith('n_') or c.startswith('p_')]
        for m in ['native', 'sr', 'bicubic']:
            for fs, cols in [('morph', MORPH), ('morph+types', MORPH + tcols)]:
                X = pd.concat([mf[m][MORPH], tabs[m].set_index('barcode').reindex(nat.index)[tcols]], axis=1)[cols]
                X = X.fillna(X.median()).fillna(0).to_numpy(float)
                for prog in PROGRAMS:
                    y = sc[prog].to_numpy(float); ok = np.isfinite(y)
                    r2 = cv_r2(X[ok], y[ok], blocks[ok]) if ok.sum() > 100 else np.nan
                    pred_rows.append(dict(meta, method=m, feature_set=fs, program=prog, cv_r2=r2, n=int(ok.sum())))
        pd.DataFrame(comp_rows).to_csv(OUT / 'composition_recovery.csv', index=False)
        pd.DataFrame(conf_rows).to_csv(OUT / 'type_confusion.csv', index=False)
        pd.DataFrame(expr_rows).to_csv(OUT / 'composition_expression.csv', index=False)
        pd.DataFrame(neo_rows).to_csv(OUT / 'neoplastic_morphology_expression.csv', index=False)
        pd.DataFrame(pred_rows).to_csv(OUT / 'prediction_cv_r2.csv', index=False)
        (OUT / 'progress.json').write_text(json.dumps(dict(done=key, elapsed_s=round(time.time() - t0))))
    (OUT / 'COMPLETE.json').write_text(json.dumps(dict(elapsed_s=round(time.time() - t0))))


if __name__ == '__main__':
    main()
