"""Step 08b — stage 2: TCGA Q70 training data (patient split + tissue tile extraction).

  Included: Q70 slides passing sr08a QC (excluding 20x slides such as M8-A5N4), 69 slides.
  Split: patient-level 80 / 10 / 10 (train / val / test), within each tissue source site (TSS code), fixed seed;
        all slides of a patient go to one set. Site composition is written to split_sites.csv.
  Tiles: up to 100 tiles of 128 µm per slide inside tissue, area-downsampled from 40x to 0.5 µm → 256 px (high-resolution target).
        At least 50% of tile pixels must have saturation > 20 (avoids background and edges).
  Validation: fixed 1.0 µm low-resolution inputs generated with the sr02 v2 degradation and a fixed seed (val_lr_1.0/), so every validation uses the same inputs.
        Training low-resolution inputs are generated on the fly and not stored.
Outputs: results/sr08/data/{split.csv, split_sites.csv, tiles.csv, hr/<split>/<tile_id>.png, val_lr_1.0/}
HEST / Xenium test sets are not used in any step here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]
from sr02_degrade_benchmark import degrade_real2, save_png  # noqa: E402

SVS = Path("/Volumes/INTENSO/TCGA_PAAD_svs")
QC = ROOT / "results" / "sr08" / "qc"
OUT = ROOT / "results" / "sr08" / "data"
TILE_UM, GT_MPP, GT_PX = 128.0, 0.5, 256
MAX_TILES = 100
SEED = 2026


def eligible() -> pd.DataFrame:
    p = pd.read_csv(QC / "tcga_qc_per_slide.csv")
    bad = (p.n_regions < 6) | (p.block_luma8_native > 2) | (p.sharpness < 0.3) | (p.mpp > 0.3)
    q = p[(p.Q == 70) & ~bad].copy()
    q["site"] = q.patient.str[5:7]
    return q


def split_patients(q: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for site, g in q.groupby("site"):
        pts = rng.permutation(sorted(g.patient.unique()))
        n = len(pts)
        n_val = int(round(0.1 * n))
        n_test = int(round(0.1 * n))
        for i, pt in enumerate(pts):
            rows.append({"patient": pt, "site": site,
                         "split": "val" if i < n_val else "test" if i < n_val + n_test else "train"})
    return pd.DataFrame(rows)


def tiles_for_slide(path: Path, mpp: float, rng) -> list[tuple[int, int, int]]:
    sl = openslide.OpenSlide(str(path))
    src = int(round(TILE_UM / mpp))
    lvl = sl.level_count - 1
    ds = sl.level_downsamples[lvl]
    th = np.asarray(sl.read_region((0, 0), lvl, sl.level_dimensions[lvl]).convert("RGB"))
    s = cv2.cvtColor(th, cv2.COLOR_RGB2HSV)[..., 1]
    _, m = cv2.threshold(cv2.GaussianBlur(s, (5, 5), 0), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(int(np.ceil(src / ds)), 1)
    gx, gy = np.meshgrid(np.arange(0, sl.dimensions[0] - src, src), np.arange(0, sl.dimensions[1] - src, src))
    cand = [(x, y) for x, y in zip(gx.ravel(), gy.ravel())
            if m[int(y / ds):int(y / ds) + k, int(x / ds):int(x / ds) + k].mean() > 0.8]
    cand = [cand[i] for i in rng.permutation(len(cand))]
    out = []
    for x, y in cand:
        img = np.asarray(sl.read_region((int(x), int(y)), 0, (src, src)).convert("RGB"))
        if (cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1] > 20).mean() < 0.5:
            continue
        out.append((int(x), int(y), src))
        if len(out) >= MAX_TILES:
            break
    sl.close()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q = eligible()
    sp = split_patients(q)
    sp.to_csv(OUT / "split.csv", index=False)
    q = q.merge(sp[["patient", "split"]], on="patient")
    sites = pd.crosstab(q.site, q.split).assign(slides=q.groupby("site").size())
    sites.to_csv(OUT / "split_sites.csv")
    print(f"{len(q)} Q70 slides, {q.patient.nunique()} patients")
    print("patients per set:", sp.split.value_counts().to_dict())
    print("slides per site (by set):")
    print(sites.to_string())

    rng = np.random.default_rng(SEED)
    rows = []
    for r in q.itertuples():
        for i, (x, y, src) in enumerate(tiles_for_slide(SVS / r.file, r.mpp, rng)):
            tid = f"{r.file[:23]}_{i:03d}"
            rows.append({"tile_id": tid, "file": r.file, "patient": r.patient, "site": r.site, "split": r.split,
                         "x": x, "y": y, "src_px": src, "mpp": r.mpp})
    tiles = pd.DataFrame(rows)
    sl_cache = {}
    vrng = np.random.default_rng(SEED + 1)
    for t in tiles.itertuples():
        if t.file not in sl_cache:
            sl_cache.clear()
            sl_cache[t.file] = openslide.OpenSlide(str(SVS / t.file))
        img = np.asarray(sl_cache[t.file].read_region((t.x, t.y), 0, (t.src_px, t.src_px)).convert("RGB"))
        hr = cv2.resize(img, (GT_PX, GT_PX), interpolation=cv2.INTER_AREA)
        save_png(OUT / "hr" / t.split / f"{t.tile_id}.png", hr)
        if t.split == "val":
            lr, _ = degrade_real2(hr, GT_PX // 2, vrng)
            save_png(OUT / "val_lr_1.0" / f"{t.tile_id}.png", lr)
    tiles.to_csv(OUT / "tiles.csv", index=False)
    print("tiles:", tiles.split.value_counts().to_dict(), "| median per slide", int(tiles.groupby("file").size().median()))


if __name__ == "__main__":
    main()
