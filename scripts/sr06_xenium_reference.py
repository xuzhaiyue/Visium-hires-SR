"""Step 06 — Xenium molecular cell typing as the patch-level reference instead of CellViT; also builds the Xenium patches whose DAPI nuclei serve as the independent nucleus reference.

Motivation: in sr04a / sr05 the "inflammatory fraction" was weakest and nearly unrelated to immune expression on GEO (ρ ≈ 0.06). The reference itself (CellViT PanNuke classes)
is unreliable, especially on frozen samples. HEST Xenium samples have transcripts and cell segmentation registered to the H&E, so each patch can be typed from molecular markers.

  make   for each Xenium sample (the only mode used in the paper):
           1. cell typing: count markers of 5 classes (immune / epithelial and parenchymal tumour / fibroblast / endothelial / smooth muscle); each class total is divided by its
              mean over all cells of the sample (corrects unequal marker numbers in the panel); take the top class; < 3 marker transcripts or top share < 0.5 = untyped
           2. sample 200 patches of 128 µm: inside tissue, >= 10 Xenium cells, at least 12 of 4×4 sub-blocks containing cells (avoids the edge of Xenium coverage)
           3. per-patch composition (class fractions of typed cells) and density; images gt (0.5 µm), real2_3.25, real2_3.8 (sr02 v2 degradation)
  eval   A. leave-one-study-out within Xenium: frozen UNI2-h + ridge regression of Xenium composition (requires sr04a, not included)
         B. control: sr04a models trained on the CellViT reference (28 HEST samples) applied to Xenium patches and compared with Xenium truth
         both use Spearman ρ within the held-out sample
  geo    train on all Xenium patches, apply to sr05 GEO spots and compare with expression scores (requires sr05, not included)
  immune split immune cells into lymphoid (T/B/NK), plasma and myeloid (only within cells already typed immune; main classes unchanged);
         leave-one-study-out within Xenium + application to GEO, each checked against the matching gene set
Outputs: results/sr_bench/xenium/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from scipy.sparse import diags as sp_diags

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from sr02_degrade_benchmark import (GT_MPP, GT_SIZE, WSI, degrade_real2, ensure_mounted, find_file,  # noqa: E402
                                    save_png)

BENCH = ROOT / "results" / "sr_bench"
OUT = BENCH / "xenium"
SAMPLES = ["TENX105", "TENX140", "TENX126", "TENX116", "TENX142", "TENX141", "TENX120", "TENX118", "TENX157",
           "TENX158", "NCBI783", "TENX199"]
N_PER_SAMPLE = 200
LR = (3.25, 3.8)
MARKERS = {
    "immune": "PTPRC CD3E CD3D CD2 CD247 CD4 CD8A IL7R CCR7 MS4A1 CD79A CD19 CD68 CD163 CD14 FCGR3A CSF1R C1QA "
              "C1QC TYROBP LYZ ITGAX NKG7 GZMB MZB1 JCHAIN CD38 IGHG1 CPA3 TPSAB1".split(),
    "epithelial": "EPCAM KRT8 KRT18 KRT19 KRT7 KRT5 KRT14 KRT17 KRT20 KRT23 CDH1 CLDN3 CLDN4 CLDN7 ELF3 ESRP1 "
                  "MUC1 CEACAM5 CEACAM6 TACSTD2 AGR2 TFF3 SPINK1 GATA3 FOXA1 ERBB2 "
                  # non-classical epithelial parenchymal/tumour cells: renal carcinoma, hepatocytes, melanocytes (PanNuke "Neoplastic")
                  "CA9 PAX8 PAX2 ALB APOA1 TTR SERPINA1 PMEL MLANA TYR SOX10".split(),
    "fibroblast": "COL1A1 COL1A2 COL3A1 COL5A1 COL6A3 DCN LUM FBLN1 FBN1 MMP2 THY1 PDGFRA PDGFRB FAP POSTN "
                  "SFRP2 SFRP4 CTHRC1".split(),
    "endothelial": "PECAM1 VWF CDH5 KDR CLDN5 EMCN PLVAP AQP1 RAMP2".split(),
    "smooth_muscle": "ACTA2 MYH11 DES CNN1 TAGLN".split(),
}
CLASSES = list(MARKERS)
IMMUNE_SUB = {   # only used to subdivide immune cells; IGKC is very high and the main source of free background, so it is not used in the main classes
    "lymphoid": "CD3E CD3D CD2 CD247 CD4 CD8A IL7R CCR7 NKG7 GZMB MS4A1 CD79A CD19".split(),
    "plasma": "MZB1 JCHAIN CD38 IGHG1 IGKC IGHA1".split(),
    "myeloid": "CD68 CD163 CD14 FCGR3A CSF1R C1QA C1QC TYROBP LYZ ITGAX".split(),
}
GEO_IMMUNE_SETS = {
    "lymphoid": ["CD3D", "CD3E", "CD2", "CD247", "IL7R", "CCR7", "MS4A1", "CD79A", "CD19"],
    "plasma": ["IGKC", "JCHAIN", "MZB1", "IGHG1", "IGHA1", "CD38"],
    "myeloid": ["CD68", "CD163", "CD14", "C1QA", "C1QB", "C1QC", "LYZ", "TYROBP", "CSF1R", "FCGR3A"],
}
SUB2GEO = {"frac_lymphoid": "lymphoid", "frac_plasma": "plasma", "frac_myeloid": "myeloid"}


# ---------- make ----------

def type_cells(sid: str) -> pd.DataFrame:
    f = OUT / f"cells_{sid}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    t = pq.read_table(find_file(f"transcripts/{sid}_transcripts.parquet"),
                      columns=["cell_id", "feature_name", "qv"]).to_pandas()
    for c in ("cell_id", "feature_name"):          # some samples (e.g. NCBI783) store bytes
        if len(t) and isinstance(t[c].iloc[0], bytes):
            t[c] = t[c].str.decode("utf-8")
    t = t[(t.cell_id != "UNASSIGNED") & (t.qv >= 20)]
    gene2cls = {g: c for c, gs in MARKERS.items() for g in gs}
    t = t[t.feature_name.isin(gene2cls)]
    cnt = pd.crosstab(t.cell_id, t.feature_name.map(gene2cls)).reindex(columns=CLASSES, fill_value=0)
    seg = pd.read_parquet(find_file(f"xenium_seg/{sid}_xenium_cell_seg.parquet"))
    cen = shapely.get_coordinates(shapely.centroid(shapely.from_wkb(seg.geometry.values)))
    cells = pd.DataFrame({"x": cen[:, 0], "y": cen[:, 1]}, index=seg.index.astype(str))
    cnt = cnt.reindex(cells.index, fill_value=0)
    norm = cnt / cnt.mean().replace(0, np.nan)                  # correct for unequal marker numbers per class
    share = norm.div(norm.sum(1), axis=0)
    cls = share.idxmax(1).where((cnt.sum(1) >= 3) & (share.max(1) >= 0.5), "unassigned")
    cells["cls"] = cls.fillna("unassigned")
    cells["n_marker_tx"] = cnt.sum(1)
    OUT.mkdir(parents=True, exist_ok=True)
    cells.reset_index(names="cell_id").to_parquet(f, index=False)
    present = {c: sum(g in set(t.feature_name) for g in gs) for c, gs in MARKERS.items()}
    print(f"  {sid}: {len(cells)} cells; markers per class in panel {present}; "
          f"type fractions {cells.cls.value_counts(normalize=True).round(2).to_dict()}")
    return cells.reset_index(names="cell_id")


def sample_patches(sid: str, wsi: WSI, mpp: float, cells: pd.DataFrame, rng) -> pd.DataFrame:
    js = json.load(open(find_file(f"tissue_seg/{sid}_contours.geojson")))
    tissue = shapely.union_all([shapely.geometry.shape(f["geometry"]) for f in js["features"]]).buffer(0)
    src = int(round(GT_SIZE * GT_MPP / mpp))
    gx, gy = np.meshgrid(np.arange(0, wsi.w - src, src), np.arange(0, wsi.h - src, src))
    gx, gy = gx.ravel(), gy.ravel()
    boxes = shapely.box(gx, gy, gx + src, gy + src)
    frac = shapely.area(shapely.intersection(boxes, tissue)) / src ** 2
    xy = cells[["x", "y"]].to_numpy()
    col, row = (xy[:, 0] // src).astype(int), (xy[:, 1] // src).astype(int)
    ncol = len(np.unique(gx))
    key = row * ncol + col
    n_cells = np.bincount(key[(col < ncol) & (key >= 0)], minlength=len(boxes))[:len(boxes)]
    sub = (np.floor((xy[:, 0] % src) / (src / 4)) + 4 * np.floor((xy[:, 1] % src) / (src / 4))).astype(int)
    occ = pd.DataFrame({"k": key, "s": sub}).drop_duplicates().groupby("k").size()
    cover = occ.reindex(np.arange(len(boxes)), fill_value=0).to_numpy()
    ok = np.flatnonzero((frac >= 0.9) & (n_cells >= 10) & (cover >= 12))
    pick = rng.choice(ok, size=min(N_PER_SAMPLE, len(ok)), replace=False)
    p = pd.DataFrame({"sample": sid, "x0": gx[pick], "y0": gy[pick], "src_px": src, "src_mpp": mpp,
                      "n_candidates": len(ok)})
    p.insert(1, "patch_id", [f"{sid}_x{i:04d}" for i in range(len(p))])
    return p


def patch_labels(p: pd.DataFrame, cells: pd.DataFrame, mpp: float) -> pd.DataFrame:
    rows = []
    for r in p.itertuples():
        c = cells[(cells.x >= r.x0) & (cells.x < r.x0 + r.src_px) & (cells.y >= r.y0) & (cells.y < r.y0 + r.src_px)]
        typed = c[c.cls != "unassigned"]
        fr = typed.cls.value_counts(normalize=True).reindex(CLASSES, fill_value=0)
        rows.append({"patch_id": r.patch_id, "n_cells": len(c), "n_typed": len(typed),
                     "density_per_mm2": len(c) / (r.src_px * mpp / 1000) ** 2, **{f"frac_{k}": v for k, v in fr.items()}})
    lab = pd.DataFrame(rows)
    lab["frac_stroma"] = lab[["frac_fibroblast", "frac_endothelial", "frac_smooth_muscle"]].sum(1)
    return lab


def make():
    ensure_mounted()
    meta = pd.read_csv("/Volumes/INTENSO/HEST1k/HEST_v1_3_0.csv").set_index("id")
    allp, alll = [], []
    for sid in SAMPLES:
        mpp = float(meta.loc[sid, "pixel_size_um_estimated"])
        cells = type_cells(sid)
        wsi = WSI(find_file(f"wsis/{sid}.tif"))
        rng = np.random.default_rng([1, int.from_bytes(sid.encode(), "little") % 2 ** 32])
        p = sample_patches(sid, wsi, mpp, cells, rng)
        lab = patch_labels(p, cells, mpp)
        d = BENCH / "patches" / sid
        for r in p.itertuples():
            gt = cv2.resize(wsi.read(r.x0, r.y0, r.src_px), (GT_SIZE, GT_SIZE), interpolation=cv2.INTER_AREA)
            save_png(d / "gt" / f"{r.patch_id}.png", gt)
            for lm in LR:
                img, _ = degrade_real2(gt, int(round(GT_SIZE * GT_MPP / lm)), rng)
                save_png(d / f"real2_{lm}" / f"{r.patch_id}.png", img)
        print(f"  {sid}: candidates {p.n_candidates.iloc[0]} → {len(p)} patches; median cells {lab.n_cells.median():.0f}, "
              f"typed {lab.n_typed.median():.0f}")
        allp.append(p.assign(dataset_title=meta.loc[sid, "dataset_title"], oncotree=meta.loc[sid, "oncotree_code"]))
        alll.append(lab)
    pd.concat(allp).to_csv(OUT / "patches_xenium.csv", index=False)
    pd.concat(alll).to_csv(OUT / "labels_xenium.csv", index=False)


# ---------- eval ----------

XEN_TARGETS = {"density_per_mm2": "cell density", "frac_immune": "immune fraction", "frac_epithelial": "epithelial/tumour fraction",
               "frac_stroma": "stromal fraction"}
# sr04a CellViT-reference targets → matching Xenium truth
CV2XEN = {"n": "density_per_mm2", "Inflammatory": "frac_immune", "Neoplastic": "frac_epithelial",
          "Connective": "frac_stroma"}
SR04A = BENCH / "sr04a"
CONDS = ["gt", "real2_3.25", "real2_3.8"]


def _ridge():
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))


def _model_getter(device):
    holder = {}

    def get():
        if "m" not in holder:
            from hemorph import context_uni2h as U
            holder["m"] = U.load_uni2h(None, device)
        return holder["m"]
    return get


def cellvit_trained(cond: str):
    """Ridge models trained on the CellViT reference over the 24 sr04a studies (one per target)."""
    ref = pd.read_csv(SR04A / "pred_real2_3.8_ridge_all.csv").set_index("patch_id")
    z = np.load(SR04A / f"emb_{cond}.npz", allow_pickle=True)
    r = ref.reindex(pd.Index(z["ids"]))
    ok = r["ref_n"].notna().to_numpy() & ~np.asarray(pd.Index(z["ids"]).str.contains("_x"))
    return {t: _ridge().fit(z["x"][ok], r[f"ref_{t}"].to_numpy()[ok]) for t in CV2XEN}


def within_rho(sample, a, b) -> pd.Series:
    from scipy.stats import spearmanr
    d = pd.DataFrame({"s": sample, "a": a, "b": b}).dropna()
    return d.groupby("s").apply(lambda g: spearmanr(g.a, g.b).statistic if len(g) > 10 else np.nan)


def evaluate(device: str, batch: int):
    from sr04a_direct_pathomics import embeddings
    import torch
    torch.set_num_threads(4)
    p = pd.read_csv(OUT / "patches_xenium.csv")
    lab = pd.read_csv(OUT / "labels_xenium.csv").set_index("patch_id").loc[p.patch_id]
    groups = p.dataset_title.to_numpy()
    get = _model_getter(device)
    rows = []
    for cond in CONDS:
        X = embeddings(cond, p, get, device, batch)
        # A: leave-one-study-out within Xenium
        for t in XEN_TARGETS:
            y = lab[t].to_numpy()
            pred = np.full(len(y), np.nan)
            for g in np.unique(groups):
                te = groups == g
                pred[te] = _ridge().fit(X[~te], y[~te]).predict(X[te])
            rows += [{"condition": cond, "target": t, "method": "trained on Xenium reference", "sample": s, "rho": v}
                     for s, v in within_rho(p["sample"], y, pred).items()]
        # B: trained on CellViT reference (28 HEST samples), predicting Xenium truth
        cvm = cellvit_trained(cond)
        for t_cv, t in CV2XEN.items():
            rows += [{"condition": cond, "target": t, "method": "trained on CellViT reference", "sample": s, "rho": v}
                     for s, v in within_rho(p["sample"], lab[t].to_numpy(), cvm[t_cv].predict(X)).items()]
    res = pd.DataFrame(rows)
    res["oncotree"] = res["sample"].map(p.drop_duplicates("sample").set_index("sample").oncotree)
    res.to_csv(OUT / "eval_rho_by_sample.csv", index=False)
    pd.set_option("display.width", 200)
    tab = res.pivot_table(index=["condition", "target"], columns="method", values="rho", aggfunc="mean")
    print("within-held-out-sample ρ (mean over 12 Xenium samples)")
    print(tab.round(2).to_string())
    print("\nreal2_3.8 per sample:")
    print(res[res.condition == "real2_3.8"].pivot_table(index=["sample", "oncotree"], columns=["target", "method"],
                                                        values="rho").round(2).to_string())


# ---------- geo ----------

def geo(device: str, batch: int):
    """Apply the Xenium-reference model (all 12 samples, real2_3.25 + real2_3.8) to the sr05 GEO spots."""
    from scipy.stats import spearmanr
    from sr04a_direct_pathomics import embeddings  # noqa: F401  (same cache format)
    import sr05_geo_molecular_validation as V
    from hemorph import context_uni2h as U
    import torch
    torch.set_num_threads(4)
    p = pd.read_csv(OUT / "patches_xenium.csv")
    lab = pd.read_csv(OUT / "labels_xenium.csv").set_index("patch_id")
    Xs, ys = [], []
    for c in ("real2_3.25", "real2_3.8"):
        z = np.load(SR04A / f"emb_{c}.npz", allow_pickle=True)
        idx = pd.Index(z["ids"]).get_indexer(p.patch_id)
        if (idx < 0).any():
            raise SystemExit(f"{c}: {int((idx < 0).sum())} Xenium patches lack UNI2-h features; run eval first")
        Xs.append(z["x"][idx])
        ys.append(lab.loc[p.patch_id, list(XEN_TARGETS)].to_numpy())
    X, Y = np.vstack(Xs), np.vstack(ys)
    models = {t: _ridge().fit(X, Y[:, j]) for j, t in enumerate(XEN_TARGETS)}
    get = _model_getter(device)
    picks = pd.read_csv(V.OUT / "picks.csv")
    rows = []
    for r in picks.itertuples():
        spots = pd.read_parquet(V.OUT / f"spots_{r.gse}.parquet")
        if spots.mpp.iloc[0] >= 5:            # outside the training scale; excluded as in sr05
            continue
        f = V.OUT / f"emb_{r.gse}.npy"
        if f.exists():
            E = np.load(f)
        else:
            S = V.load_sample(r)
            size = int(round(V.CROP_UM / S["mpp"]))
            tiles = [cv2.resize(U.crop(S["img"], x, y, size, np.array([255, 255, 255], np.uint8)), (224, 224),
                                interpolation=cv2.INTER_CUBIC) for x, y in zip(spots.x, spots.y)]
            E = U.embed(get(), tiles, device, batch)
            np.save(f, E)
        for t, m in models.items():
            spots[f"xen_{t}"] = m.predict(E)
        spots.to_parquet(OUT / f"geo_spots_{r.gse}.parquet", index=False)
        pairs = {"immune": ("xen_frac_immune", "pred_Inflammatory"), "epithelial": ("xen_frac_epithelial", "pred_Neoplastic"),
                 "stroma": ("xen_frac_stroma", "pred_Connective")}
        for score, (fx, fc) in pairs.items():
            for name, col in (("trained on Xenium reference", fx), ("trained on CellViT reference", fc), ("haematoxylin intensity", "H_od")):
                rho = spearmanr(spots[col], spots[score], nan_policy="omit").statistic
                rows.append({"gse": r.gse, "score": score, "method": name, "rho": rho if name != "haematoxylin intensity" or score == "epithelial"
                             else -rho if score == "stroma" else rho})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "geo_rho_by_sample.csv", index=False)
    pd.set_option("display.width", 200)
    print("GEO spots: prediction vs expression score (haematoxylin sign flipped for stroma so the expected direction is positive)")
    print(res.pivot_table(index="gse", columns=["score", "method"], values="rho").round(2).to_string())
    print("\nmean")
    print(res.pivot_table(index="score", columns="method", values="rho", aggfunc="mean").round(2).to_string())


# ---------- immune ----------

def immune_subtypes(sid: str) -> pd.Series:
    f = OUT / f"immune_sub_{sid}.parquet"
    if f.exists():
        return pd.read_parquet(f).set_index("cell_id")["sub"]
    cells = type_cells(sid).set_index("cell_id")
    imm = cells.index[cells.cls == "immune"]
    t = pq.read_table(find_file(f"transcripts/{sid}_transcripts.parquet"),
                      columns=["cell_id", "feature_name", "qv"]).to_pandas()
    for c in ("cell_id", "feature_name"):
        if len(t) and isinstance(t[c].iloc[0], bytes):
            t[c] = t[c].str.decode("utf-8")
    g2s = {g: k for k, gs in IMMUNE_SUB.items() for g in gs}
    t = t[(t.qv >= 20) & t.cell_id.isin(imm) & t.feature_name.isin(g2s)]
    cnt = pd.crosstab(t.cell_id, t.feature_name.map(g2s)).reindex(index=imm, columns=list(IMMUNE_SUB), fill_value=0)
    norm = cnt / cnt.mean().replace(0, np.nan)
    share = norm.div(norm.sum(1), axis=0)
    sub = share.idxmax(1).where((cnt.sum(1) >= 2) & (share.max(1) >= 0.5), "immune_other").fillna("immune_other")
    pd.DataFrame({"cell_id": imm, "sub": sub.to_numpy()}).to_parquet(f, index=False)
    present = {k: sum(g in set(t.feature_name) for g in gs) for k, gs in IMMUNE_SUB.items()}
    print(f"  {sid}: immune cells {len(imm)}; markers in panel {present}; subtypes {sub.value_counts(normalize=True).round(2).to_dict()}")
    return sub


def immune(device: str, batch: int):
    from scipy.stats import spearmanr
    import sr05_geo_molecular_validation as V
    p = pd.read_csv(OUT / "patches_xenium.csv")
    rows = []
    for sid, ps in p.groupby("sample", sort=False):
        cells = type_cells(sid).set_index("cell_id")
        cells["sub"] = immune_subtypes(sid).reindex(cells.index)
        typed = cells[cells.cls != "unassigned"]
        for r in ps.itertuples():
            c = typed[(typed.x >= r.x0) & (typed.x < r.x0 + r.src_px) & (typed.y >= r.y0) & (typed.y < r.y0 + r.src_px)]
            n = max(len(c), 1)
            rows.append({"patch_id": r.patch_id, **{f"frac_{k}": (c["sub"] == k).sum() / n for k in IMMUNE_SUB}})
    lab = pd.DataFrame(rows).set_index("patch_id").loc[p.patch_id]
    lab.to_csv(OUT / "labels_immune_sub.csv")
    groups = p.dataset_title.to_numpy()
    res = []
    Xg = {}
    for cond in ("real2_3.25", "real2_3.8"):
        z = np.load(SR04A / f"emb_{cond}.npz", allow_pickle=True)
        idx = pd.Index(z["ids"]).get_indexer(p.patch_id)
        assert (idx >= 0).all()
        X = z["x"][idx]
        Xg[cond] = X
        for t in SUB2GEO:
            y = lab[t].to_numpy()
            pred = np.full(len(y), np.nan)
            for g in np.unique(groups):
                te = groups == g
                pred[te] = _ridge().fit(X[~te], y[~te]).predict(X[te])
            res += [{"condition": cond, "target": t, "sample": s, "rho": v}
                    for s, v in within_rho(p["sample"], y, pred).items()]
    res = pd.DataFrame(res)
    res.to_csv(OUT / "immune_sub_eval_rho.csv", index=False)
    pd.set_option("display.width", 200)
    print("Within Xenium (leave-one-study-out, ρ within held-out sample)")
    print(res.pivot_table(index="sample", columns=["condition", "target"], values="rho").round(2).to_string())
    print(res.groupby(["condition", "target"]).rho.mean().round(2).to_string())
    print("median immune-subtype fraction of typed cells per sample:")
    print(lab.assign(sample=p.set_index("patch_id").loc[lab.index, "sample"].to_numpy())
          .groupby("sample").median().round(3).to_string())

    # GEO: train on all Xenium patches (both resolutions) → GEO spots; check each of three immune gene sets
    X = np.vstack([Xg["real2_3.25"], Xg["real2_3.8"]])
    Y = np.vstack([lab.to_numpy(), lab.to_numpy()])
    cols = list(lab.columns)
    models = {t: _ridge().fit(X, Y[:, cols.index(t)]) for t in SUB2GEO}
    picks = pd.read_csv(V.OUT / "picks.csv")
    geo_rows = []
    for r in picks.itertuples():
        spots = pd.read_parquet(V.OUT / f"spots_{r.gse}.parquet")
        if spots.mpp.iloc[0] >= 5:
            continue
        E = np.load(V.OUT / f"emb_{r.gse}.npy")
        S = V.load_sample(r)
        assert len(S["pos"]) == len(spots) and (S["pos"].index == spots.barcode).all()
        tot = np.asarray(S["X"].sum(1)).ravel()
        lg = (sp_diags(1e4 / np.maximum(tot, 1)) @ S["X"]).log1p()
        first = {g: i for i, g in reversed(list(enumerate(S["genes"])))}
        for t, gname in SUB2GEO.items():
            idx = [first[g] for g in GEO_IMMUNE_SETS[gname] if g in first]
            score = np.asarray(lg[:, idx].mean(1)).ravel()
            pred = models[t].predict(E)
            geo_rows.append({"gse": r.gse, "subset": gname, "n_genes": len(idx),
                             "rho_pred": spearmanr(pred, score).statistic,
                             "rho_H_od": spearmanr(spots.H_od, score).statistic,
                             "score_median": float(np.median(score))})
    g = pd.DataFrame(geo_rows)
    g.to_csv(OUT / "immune_sub_geo_rho.csv", index=False)
    print("\nGEO: predicted immune-subtype fraction vs matching gene-set score")
    print(g.pivot_table(index="gse", columns="subset", values=["rho_pred", "rho_H_od"]).round(2).to_string())
    print(g.groupby("subset")[["rho_pred", "rho_H_od"]].mean().round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["make", "eval", "geo", "immune"])
    ap.add_argument("--samples", nargs="*", help="only these samples (debugging)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    if args.samples:
        SAMPLES[:] = args.samples
    if args.step == "make":
        make()
    elif args.step == "eval":
        evaluate(args.device, args.batch)
    elif args.step == "geo":
        geo(args.device, args.batch)
    else:
        immune(args.device, args.batch)


if __name__ == "__main__":
    main()
