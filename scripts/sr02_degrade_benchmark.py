"""Step 02 — degradation benchmark: sample fixed patches from HEST WSIs and generate 0.5 µm targets plus several low-resolution versions.

Outputs (results/sr_bench/):
  patches.csv                 sample, level-0 coordinates, nucleus count and tissue fraction of each patch
  degrade_params.csv          random parameters of each "real" degraded patch (reproducible)
  gt_nuclei/<sample>.parquet  HEST CellViT nuclei (converted to 0.5 µm patch coordinates) as the morphology reference
  patches/<sample>/native/        native resolution resampled to 0.25 µm/px, 512×512 (CellViT++ x40 input scale) — segmentation reference
  patches/<sample>/gt/            0.5 µm/px, 256×256 (128 µm field) — first "degraded" level and source of the lower levels
  patches/<sample>/clean_<mpp>/   area downsampling only
  patches/<sample>/real_<mpp>/    v1 (deprecated, kept for old results): stain perturbation + fading + blur/defocus + downsampling + noise + JPEG on the low-res image
  patches/<sample>/real2_<mpp>/   v2 (after sr03c/sr03d checks): stain perturbation (no global fading) + blur/defocus + JPEG at 0.5 µm
                                  → downsampling + noise. Real hires PNGs show no block artefacts and no systematic fading
                                  (results/sr_survey/hires_calibration/README.md)
  degrade_params_v2.csv           v2 random parameters per patch (v1 degrade_params.csv is kept; sr03c depends on it)
  preview/<sample>.png        one patch per sample at every level, for visual inspection

All methods (bicubic / SwinIR / direct regression) are compared on the same patches.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shapely
import tifffile
import zarr

ROOT = Path(__file__).resolve().parents[1]
HEST_ROOTS = [Path("/Volumes/INTENSO/HEST1k"), Path("/Volumes/INTENSO_BIG/HEST1k")]
SPARSEBUNDLE = Path("/Volumes/INTENSO/bigfiles.sparsebundle")  # FAT32 cannot hold files >4 GiB; large files live in this APFS image
OUT = ROOT / "results" / "sr_bench"

GT_MPP = 0.5
NATIVE_MPP = 0.25                  # CellViT++ x40 model scale
GT_SIZE = 256                      # 256 px × 0.5 µm = 128 µm
LR_MPPS = (1.0, 2.0, 3.25, 3.8)    # 3.25 µm ≈ 6.5 mm / 2000 px; 3.8 µm = measured GEO hires median (sr03d)
MIN_TISSUE = 0.9
MIN_NUCLEI = 10

# Ruifrok & Johnston H&E-DAB stain matrix (rows: OD vectors of H, E, DAB)
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
RGB_FROM_HED /= np.linalg.norm(RGB_FROM_HED, axis=1, keepdims=True)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)


def ensure_mounted():
    if not HEST_ROOTS[1].parent.exists() and SPARSEBUNDLE.exists():
        subprocess.run(["hdiutil", "attach", str(SPARSEBUNDLE)], check=True, capture_output=True)


def find_file(rel: str) -> Path:
    for r in HEST_ROOTS:
        if (r / rel).exists():
            return r / rel
    raise FileNotFoundError(rel)


class WSI:
    """Level 0 only; read windows through zarr without loading the whole image."""

    def __init__(self, path: Path):
        self.tif = tifffile.TiffFile(path)
        self.arr = zarr.open(self.tif.pages[0].aszarr(), mode="r")
        self.h, self.w = self.arr.shape[:2]

    def read(self, x0: int, y0: int, size: int) -> np.ndarray:
        return np.asarray(self.arr[y0:y0 + size, x0:x0 + size, :3])


# ---------- patch sampling ----------

def sample_patches(sid: str, wsi: WSI, mpp: float, nuc_centroids: np.ndarray, n: int, rng) -> pd.DataFrame:
    js = json.load(open(find_file(f"tissue_seg/{sid}_contours.geojson")))
    tissue = shapely.union_all([shapely.geometry.shape(f["geometry"]) for f in js["features"]]).buffer(0)
    src = int(round(GT_SIZE * GT_MPP / mpp))
    xs = np.arange(0, wsi.w - src, src)
    ys = np.arange(0, wsi.h - src, src)
    gx, gy = np.meshgrid(xs, ys)
    boxes = shapely.box(gx.ravel(), gy.ravel(), gx.ravel() + src, gy.ravel() + src)
    frac = shapely.area(shapely.intersection(boxes, tissue)) / src ** 2
    tree = shapely.STRtree(shapely.points(nuc_centroids))
    n_nuc = np.bincount(tree.query(boxes, predicate="contains")[0], minlength=len(boxes))
    ok = np.flatnonzero((frac >= MIN_TISSUE) & (n_nuc >= MIN_NUCLEI))
    pick = rng.choice(ok, size=min(n, len(ok)), replace=False)
    return pd.DataFrame({"sample": sid, "x0": gx.ravel()[pick], "y0": gy.ravel()[pick], "src_px": src,
                         "src_mpp": mpp, "tissue_frac": frac[pick], "n_nuclei": n_nuc[pick],
                         "n_candidates": len(ok)})


def gt_nuclei(nuc: pd.DataFrame, geoms, patches: pd.DataFrame, mpp: float) -> pd.DataFrame:
    """Convert HEST CellViT nuclei with centroids inside the patch to 0.5 µm patch coordinates."""
    tree = shapely.STRtree(shapely.centroid(geoms))
    s = mpp / GT_MPP
    rows = []
    for _, p in patches.iterrows():
        idx = tree.query(shapely.box(p.x0, p.y0, p.x0 + p.src_px, p.y0 + p.src_px), predicate="contains")
        g = shapely.transform(geoms[idx], lambda c: (c - [p.x0, p.y0]) * s)
        c = shapely.get_coordinates(shapely.centroid(g))
        rows.append(pd.DataFrame({"patch_id": p.patch_id, "cell_id": nuc.cell_id.values[idx],
                                  "class": nuc["class"].values[idx], "cx": c[:, 0], "cy": c[:, 1],
                                  "area_um2": shapely.area(geoms[idx]) * mpp ** 2,
                                  "wkb": shapely.to_wkb(g)}))
    return pd.concat(rows, ignore_index=True)


# ---------- degradation ----------

def stain_jitter(img: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    """HED-space perturbation: per-channel OD scale + shift; global fading mimics pale / frozen sections."""
    od = -np.log(np.clip(img.astype(np.float32) / 255, 1e-6, 1))
    hed = od.reshape(-1, 3) @ HED_FROM_RGB
    alpha = rng.uniform(0.85, 1.1, 3)
    beta = rng.uniform(-0.03, 0.03, 3)
    fade = rng.uniform(0.7, 1.0)
    hed = (hed * alpha + beta) * fade
    out = np.exp(-(hed @ RGB_FROM_HED)).reshape(img.shape) * 255
    return np.clip(out, 0, 255).astype(np.uint8), {"stain_alpha": alpha.round(3).tolist(),
                                                    "stain_beta": beta.round(3).tolist(), "fade": round(fade, 3)}


def degrade_real(gt: np.ndarray, lr_size: int, rng) -> tuple[np.ndarray, dict]:
    img, prm = stain_jitter(gt, rng)
    # optical PSF: Gaussian (µm scale, converted to 0.5 µm pixels); defocus disc added with probability 0.3
    sigma_um = rng.uniform(0.3, 1.2)
    img = cv2.GaussianBlur(img, (0, 0), sigma_um / GT_MPP)
    prm["blur_sigma_um"] = round(sigma_um, 3)
    if rng.random() < 0.3:
        r = max(1, int(round(rng.uniform(0.5, 1.5) / GT_MPP)))
        k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        cv2.circle(k, (r, r), r, 1, -1)
        img = cv2.filter2D(img, -1, k / k.sum())
        prm["defocus_r_um"] = r * GT_MPP
    img = cv2.resize(img, (lr_size, lr_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    noise = rng.uniform(1, 5)
    img += rng.normal(0, noise, img.shape)
    prm["noise_sd"] = round(noise, 2)
    q = int(rng.integers(60, 96))
    ok, buf = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    prm["jpeg_q"] = q
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1], prm


def degrade_real2(gt: np.ndarray, lr_size: int, rng) -> tuple[np.ndarray, dict]:
    """v2: no global fading; JPEG moved before downsampling (real order: scanner stores JPEG → Space Ranger downsamples to PNG).
    Blur, defocus and noise ranges are those of v1 and are not calibrated."""
    od = -np.log(np.clip(gt.astype(np.float32) / 255, 1e-6, 1))
    alpha, beta = rng.uniform(0.85, 1.1, 3), rng.uniform(-0.03, 0.03, 3)
    hed = od.reshape(-1, 3) @ HED_FROM_RGB * alpha + beta
    img = np.clip(np.exp(-(hed @ RGB_FROM_HED)).reshape(gt.shape) * 255, 0, 255).astype(np.uint8)
    prm = {"stain_alpha": alpha.round(3).tolist(), "stain_beta": beta.round(3).tolist()}
    sigma_um = rng.uniform(0.3, 1.2)
    img = cv2.GaussianBlur(img, (0, 0), sigma_um / GT_MPP)
    prm["blur_sigma_um"] = round(sigma_um, 3)
    if rng.random() < 0.3:
        r = max(1, int(round(rng.uniform(0.5, 1.5) / GT_MPP)))
        k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        cv2.circle(k, (r, r), r, 1, -1)
        img = cv2.filter2D(img, -1, k / k.sum())
        prm["defocus_r_um"] = r * GT_MPP
    q = int(rng.integers(60, 96))
    ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    prm["jpeg_q_at_0.5um"] = q
    img = cv2.resize(img, (lr_size, lr_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    noise = rng.uniform(1, 5)
    img += rng.normal(0, noise, img.shape)
    prm["noise_sd"] = round(noise, 2)
    return np.clip(img, 0, 255).astype(np.uint8), prm


def save_png(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgb[..., ::-1])


def preview(sid: str, imgs: dict[str, np.ndarray]):
    tiles = [cv2.resize(v, (GT_SIZE, GT_SIZE), interpolation=cv2.INTER_NEAREST) for v in imgs.values()]
    bar = np.full((24, GT_SIZE * len(tiles), 3), 255, np.uint8)
    for i, k in enumerate(imgs):
        cv2.putText(bar, k, (i * GT_SIZE + 4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    save_png(OUT / "preview" / f"{sid}.png", np.vstack([bar, np.hstack(tiles)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(ROOT / "results/sr_survey/hest_pilot_samples.csv"))
    ap.add_argument("--n-per-sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ensure_mounted()
    meta = pd.read_csv(args.samples)
    all_p, all_prm = [], []
    for _, m in meta.iterrows():
        sid, mpp = m.id, float(m.px)
        rng = np.random.default_rng([args.seed, int.from_bytes(sid.encode(), "little") % 2 ** 32])
        wsi = WSI(find_file(f"wsis/{sid}.tif"))
        nuc = pd.read_parquet(find_file(f"cellvit_seg/{sid}_cellvit_seg.parquet"))
        geoms = shapely.from_wkb(nuc.geometry.values)
        cents = shapely.get_coordinates(shapely.centroid(geoms))
        p = sample_patches(sid, wsi, mpp, cents, args.n_per_sample, rng)
        p.insert(1, "patch_id", [f"{sid}_{i:04d}" for i in range(len(p))])
        print(f"== {sid}  mpp={mpp:.3f}  {wsi.w}x{wsi.h}  candidates {p.n_candidates.iloc[0]} → sampled {len(p)}")

        (OUT / "gt_nuclei").mkdir(parents=True, exist_ok=True)
        gt_nuclei(nuc, geoms, p, mpp).to_parquet(OUT / "gt_nuclei" / f"{sid}.parquet", index=False)

        d = OUT / "patches" / sid
        for i, r in enumerate(p.itertuples()):
            src = wsi.read(r.x0, r.y0, r.src_px)
            nat_size = int(round(GT_SIZE * GT_MPP / NATIVE_MPP))
            save_png(d / "native" / f"{r.patch_id}.png",
                     cv2.resize(src, (nat_size, nat_size),
                                interpolation=cv2.INTER_AREA if r.src_px > nat_size else cv2.INTER_CUBIC))
            gt = cv2.resize(src, (GT_SIZE, GT_SIZE), interpolation=cv2.INTER_AREA)
            save_png(d / "gt" / f"{r.patch_id}.png", gt)
            show = {"gt_0.5": gt}
            for lm in LR_MPPS:
                n = int(round(GT_SIZE * GT_MPP / lm))
                clean = cv2.resize(gt, (n, n), interpolation=cv2.INTER_AREA)
                real, prm = degrade_real2(gt, n, rng)
                save_png(d / f"clean_{lm}" / f"{r.patch_id}.png", clean)
                save_png(d / f"real2_{lm}" / f"{r.patch_id}.png", real)
                all_prm.append({"patch_id": r.patch_id, "lr_mpp": lm, "lr_px": n,
                                "eff_mpp": GT_SIZE * GT_MPP / n, **{k: json.dumps(v) if isinstance(v, list) else v
                                                                    for k, v in prm.items()}})
                show[f"clean_{lm}"], show[f"real2_{lm}"] = clean, real
            if i == 0:
                preview(sid, show)
        all_p.append(p)
    pd.concat(all_p).to_csv(OUT / "patches.csv", index=False)
    pd.DataFrame(all_prm).to_csv(OUT / "degrade_params_v2.csv", index=False)
    print("done:", OUT)


if __name__ == "__main__":
    main()
