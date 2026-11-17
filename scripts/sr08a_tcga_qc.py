"""Step 08a — quality control of TCGA-PAAD training data (stage 1 of the sr08 plan).

TCGA slides are stored with JPEG Q=30 (139 slides) or Q=70 (70 slides). Rule: quantify and visually inspect native 40x tiles to judge whether JPEG blocking affects
nuclear boundaries and chromatin; exclude only slides that are severely affected; do not deblock TCGA and then call it ground truth.

Per slide: find tissue with Otsu on saturation in the thumbnail, take 12 random 512×512 regions (40x, about 128 µm; top-left aligned to 16 px so the JPEG grid is preserved) and measure:
  block_luma8_native     blocking on the 8 px luma grid of the native image (1 = none)
  block_chroma16_native  blocking on the 16 px chroma grid of the native image (4:2:0 chroma blocks)
  block_luma4_gt         blocking left on the 4 px grid (= native 8 px) after area downsampling to 0.5 µm (training target)
  sharpness              Laplacian variance / luma variance at 0.5 µm (low = out of focus)
Also writes an inspection montage: Q30 and Q70 slides ordered by blocking, each with one native 40x tile and the matching 0.5 µm tile.
Outputs: results/sr08/qc/
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
from sr03d_geo_hires_calibration import blockiness  # noqa: E402

SVS = Path("/Volumes/INTENSO/TCGA_PAAD_svs")
OUT = ROOT / "results" / "sr08" / "qc"
N_REGIONS = 12
SIZE = 512
GT_MPP = 0.5


def tissue_regions(sl: openslide.OpenSlide, rng) -> list[tuple[int, int]]:
    lvl = sl.level_count - 1
    th = np.asarray(sl.read_region((0, 0), lvl, sl.level_dimensions[lvl]).convert("RGB"))
    s = cv2.cvtColor(th, cv2.COLOR_RGB2HSV)[..., 1]
    _, m = cv2.threshold(cv2.GaussianBlur(s, (5, 5), 0), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ds = sl.level_downsamples[lvl]
    k = max(int(np.ceil(SIZE / ds)), 1)
    m = cv2.erode(m, np.ones((k, k), np.uint8))
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return []
    pick = rng.choice(len(ys), size=min(N_REGIONS * 3, len(ys)), replace=False)
    out = []
    for i in pick:
        x, y = int(xs[i] * ds) // 16 * 16, int(ys[i] * ds) // 16 * 16
        if x + SIZE < sl.dimensions[0] and y + SIZE < sl.dimensions[1]:
            out.append((x, y))
    return out


def measure(img: np.ndarray, mpp: float) -> dict:
    ycc = cv2.cvtColor(img[..., ::-1], cv2.COLOR_BGR2YCrCb).astype(np.float32)
    n = int(round(SIZE * mpp / GT_MPP))
    gt = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
    y_gt = cv2.cvtColor(gt[..., ::-1], cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    lap = cv2.Laplacian(y_gt, cv2.CV_32F)
    return {"block_luma8_native": blockiness(ycc[..., 0], 8),
            "block_chroma16_native": float(np.mean([blockiness(ycc[..., 1], 16), blockiness(ycc[..., 2], 16)])),
            "block_luma4_gt": blockiness(y_gt, 4), "sharpness": float(lap.var() / max(y_gt.var(), 1e-6)),
            "tissue_sat": float(cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1].mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(OUT / "tcga_inventory.csv")
    rng = np.random.default_rng(0)
    rows, keep_tiles = [], {}
    for r in inv.itertuples():
        sl = openslide.OpenSlide(str(SVS / r.file))
        regs = []
        for x, y in tissue_regions(sl, rng):
            img = np.asarray(sl.read_region((x, y), 0, (SIZE, SIZE)).convert("RGB"))
            if cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1].mean() < 25:      # mostly background
                continue
            regs.append({"file": r.file, "x": x, "y": y, **measure(img, r.mpp)})
            if len(regs) == 1:
                keep_tiles[r.file] = img
            if len(regs) >= N_REGIONS:
                break
        rows += regs
        sl.close()
    reg = pd.DataFrame(rows)
    reg.to_csv(OUT / "tcga_qc_regions.csv", index=False)
    per = reg.groupby("file").median(numeric_only=True).drop(columns=["x", "y"]).join(
        inv.set_index("file")[["patient", "Q", "mpp", "scanner"]])
    per["n_regions"] = reg.groupby("file").size()
    per.to_csv(OUT / "tcga_qc_per_slide.csv")
    pd.set_option("display.width", 200)
    print(per.groupby("Q")[["block_luma8_native", "block_chroma16_native", "block_luma4_gt", "sharpness"]]
          .describe(percentiles=[.1, .5, .9]).round(3).T.to_string())

    # inspection montage: Q30 worst / median / best and Q70 median slides; each: central 256 px of the native 40x tile + the matching 0.5 µm tile
    q30 = per[per.Q == 30].sort_values("block_luma8_native")
    q70 = per[per.Q == 70].sort_values("block_luma8_native")
    pick = [("Q30 worst", q30.index[-1]), ("Q30 worst", q30.index[-2]), ("Q30 median", q30.index[len(q30) // 2]),
            ("Q30 median", q30.index[len(q30) // 2 + 1]), ("Q30 best", q30.index[0]), ("Q70 median", q70.index[len(q70) // 2]),
            ("Q70 median", q70.index[len(q70) // 2 + 1])]
    cols = []
    for lab, f in pick:
        img = keep_tiles[f]
        c = img[128:384, 128:384]
        native = cv2.resize(c, (512, 512), interpolation=cv2.INTER_NEAREST)        # 40x, enlarged 2x to show pixels
        gt = cv2.resize(cv2.resize(c, (128, 128), interpolation=cv2.INTER_AREA), (512, 512), interpolation=cv2.INTER_NEAREST)
        head = np.full((28, 512, 3), 255, np.uint8)
        cv2.putText(head, f"{lab}  {f[:23]}  b8={per.loc[f, 'block_luma8_native']:.2f}", (4, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cols.append(np.vstack([head, native, np.full((6, 512, 3), 255, np.uint8), gt]))
    cv2.imwrite(str(OUT / "qc_montage.png"), np.hstack(cols)[..., ::-1])
    print("montage:", OUT / "qc_montage.png")


if __name__ == "__main__":
    main()
