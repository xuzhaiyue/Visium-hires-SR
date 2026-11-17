"""Step 08h — hand-traced nuclear contours vs predicted contours of each method (plan x4 §4).

Hand contours (results/sr08/contours/raw) are drawn in native crop coordinates (0.25 µm/px); adding the crop origin gives patch coordinates on the same scale as
the CellViT contours on the mosaic. For each hand contour, the predicted nucleus with the nearest centroid (≤ 3 µm) in each condition is taken:
  detected    whether the condition has a predicted nucleus within 3 µm
  IoU         intersection over union of the hand and predicted polygons
  area        |predicted area / hand area − 1|
  circularity |predicted − hand circularity|; freehand lines are jagged and lower circularity, so both contours are resampled at equal spacing (0.25 µm)
              and lightly smoothed (σ = 0.5 µm along the perimeter) before computing 4πA / P²
Conditions: native (CellViT on the native image, upper bound), bicubic and SwinIR (×2: 1.0 µm input; ×4: 2.0 µm input).
SwinIR vs bicubic paired Wilcoxon on nuclei detected by both. Outputs: results/sr08/contours/eval/
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from hemorph import cellvit_io as CV  # noqa: E402

BENCH = ROOT / "results" / "sr_bench"
C = ROOT / "results" / "sr08" / "contours"
OUT = C / "eval"
MPP = 0.25
MATCH_PX = 3.0 / MPP
CONDS = {"native (upper bound)": "native", "bicubic ×2": "real2_1.0", "SwinIR ×2": "swinir2_real2_1.0",
         "bicubic ×4": "real2_2.0", "SwinIR ×4": "swinir4_real2_2.0"}


def smooth_metrics(xy: np.ndarray) -> tuple[float, float]:
    """(area µm², circularity) after equal-spacing resampling and Gaussian smoothing along the perimeter."""
    ring = np.vstack([xy, xy[:1]])
    seg = np.hypot(*np.diff(ring, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    n = max(int(L / 1.0), 12)                                  # 1 px = 0.25 µm
    t = np.linspace(0, L, n, endpoint=False)
    rs = np.column_stack([np.interp(t, s, ring[:, 0]), np.interp(t, s, ring[:, 1])])
    rs = np.column_stack([gaussian_filter1d(rs[:, k], 2.0, mode="wrap") for k in range(2)])
    poly = shapely.Polygon(rs).buffer(0)
    a, p = poly.area, poly.length
    return a * MPP ** 2, 4 * np.pi * a / p ** 2 if p > 0 else np.nan


def predicted(method: str, cond: str, pids: set[str]) -> dict[str, list[np.ndarray]]:
    d = BENCH / "mosaic" / f"{method}_{cond}"
    if not any(d.rglob("cells.json")):
        return {}
    cells, conts = CV.load_cells(d / f"cellvit_{method}_{MPP:g}mpp")
    lay = pd.read_csv(d / "layout.csv").set_index("patch_id")
    lay = lay[lay.index.isin(pids)]
    out = {}
    cx, cy = cells.cx.to_numpy(), cells.cy.to_numpy()
    for pid, r in lay.iterrows():
        k = np.flatnonzero((cx >= r.ox) & (cx < r.ox + 512) & (cy >= r.oy) & (cy < r.oy + 512))
        out[pid] = [conts[i] - [r.ox, r.oy] for i in k]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.DataFrame([{"id": Path(f).stem, **json.load(open(f))} for f in glob.glob(str(C / "raw/contours/*.json"))])
    k = pd.read_csv(C / "key.csv").merge(lab, on="id")
    drawn = k[k.status == "drawn"].reset_index(drop=True)
    pids = set(drawn.patch_id)
    rows = []
    for name, cond in CONDS.items():
        preds = {}
        for m in ("bicubic", "bicubic2"):
            preds.update(predicted(m, cond, pids))
        if not preds:
            print(f"skip {name}: no results yet")
            continue
        for r in drawn.itertuples():
            if r.patch_id not in preds:            # condition does not cover this patch (e.g. pilot samples only); not counted
                continue
            hxy = np.asarray(r.pts, float) + [r.crop_x0, r.crop_y0]
            hpoly = shapely.Polygon(hxy).buffer(0)
            h_area, h_circ = smooth_metrics(hxy)
            hc = np.array(hpoly.centroid.coords[0])
            best, bd = None, np.inf
            for c in preds.get(r.patch_id, []):
                dd = np.hypot(*(c.mean(0) - hc))
                if dd < bd:
                    best, bd = c, dd
            row = {"id": r.id, "cat": r.cat, "sample": r.sample, "method": name, "human_area_um2": h_area,
                   "human_circ": h_circ, "detected": best is not None and bd <= MATCH_PX}
            if row["detected"]:
                ppoly = shapely.Polygon(best).buffer(0)
                p_area, p_circ = smooth_metrics(best)
                row.update(iou=hpoly.intersection(ppoly).area / hpoly.union(ppoly).area, pred_area_um2=p_area,
                           area_err=abs(p_area / h_area - 1), area_bias=p_area / h_area - 1, circ_err=abs(p_circ - h_circ),
                           centroid_um=bd * MPP)
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "per_nucleus.csv", index=False)
    pd.set_option("display.width", 200)
    g = res.groupby("method", sort=False)
    summ = pd.DataFrame({"detection rate": g.detected.mean(), "median IoU": g.iou.median(), "median area error": g.area_err.median(),
                         "median area bias": g.area_bias.median(), "median circularity error": g.circ_err.median(), "n detected": g.detected.sum()})
    print(f"hand-traced nuclei: {len(drawn)} ({drawn['sample'].nunique()} samples)")
    print(summ.round(3).to_string())
    print("\nhand-traced nuclei covered per condition:", res.groupby("method", sort=False).size().to_dict())
    print("\nby category: detection rate / median IoU / median area error")
    for m in ("detected", "iou", "area_err"):
        print(f"-- {m}")
        print(res.pivot_table(index="cat", columns="method", values=m, aggfunc="mean" if m == "detected" else "median",
                              sort=False).round(2).to_string())
    print("\ndetection rate by category:")
    print(res.pivot_table(index="cat", columns="method", values="detected", aggfunc="mean", sort=False).round(2).to_string())

    from scipy.stats import wilcoxon
    lines = []
    for sc in ("×2", "×4"):
        a, b = f"bicubic {sc}", f"SwinIR {sc}"
        if a not in set(res.method) or b not in set(res.method):
            continue
        w = res[res.method.isin([a, b])].pivot(index="id", columns="method", values=["iou", "area_err", "circ_err"])
        w = w.dropna()
        lines.append(f"\n{sc}: nuclei detected by both methods n = {len(w)}")
        for m, lab_ in (("iou", "IoU"), ("area_err", "area error"), ("circ_err", "circularity error")):
            d = w[(m, b)] - w[(m, a)]
            p = wilcoxon(d).pvalue if len(d) >= 5 else np.nan
            lines.append(f"  {lab_}: bicubic {w[(m, a)].median():.3f} → SwinIR {w[(m, b)].median():.3f}"
                         f" (median paired difference {d.median():+.3f}, p = {p:.3g})")
    print("\n".join(lines))
    (OUT / "summary.txt").write_text(summ.round(3).to_string() + "\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
