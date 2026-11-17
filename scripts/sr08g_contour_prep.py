"""Step 08g — items for hand-traced nuclear contours (plan x4 §4).

Only the primary set (13 HEST samples with native ≤ 0.3 µm), where nuclear boundaries are clear. Each item is a 32 µm crop of the native image (0.25 µm/px) centred on the target nucleus
(128 px, shown enlarged 4x for tracing); only the native image is shown and the annotator traces the central nucleus blinded to method.
Categories (defined by ×2 detections; contours are drawn on the native image, so ×4 and later methods can be evaluated on the same nuclei):
  both      30  native reference nuclei detected by both SwinIR×2 and bicubic
  sr_only   20  native reference nuclei detected only by SwinIR×2
  bic_only  10  native reference nuclei detected only by bicubic
  missed    15  native reference nuclei missed by both methods
  sr_unique 15  SwinIR×2-only points judged "nucleus" in labelling round 1 (absent from the reference)
The first four categories are stratified by reference nuclear-area tertile; round-robin over samples. The page shows only item IDs and images; categories and coordinates are kept in the local key.csv.
Outputs: results/sr08/contours/{items.json, key.csv}
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from hemorph import nucleus_match as NM  # noqa: E402
from sr03b_eval_morphology import load_condition  # noqa: E402

BENCH = ROOT / "results" / "sr_bench"
OUT = ROOT / "results" / "sr08" / "contours"
CROP = 128                       # 32 µm @ 0.25 µm
MARGIN = CROP // 2
SEED = 23
N = {"both": 30, "sr_only": 20, "bic_only": 10, "missed": 15, "sr_unique": 15}


def ref_categories() -> pd.DataFrame:
    meta = pd.read_csv(ROOT / "results/sr_survey/hest_benchmark_candidates.csv").set_index("id")
    hq = set(meta.index[meta.px <= 0.3])
    rows = []
    for m in ("bicubic", "bicubic2"):
        ref = load_condition(m, "native")
        ref = ref[ref["sample"].isin(hq)]
        if ref.empty:
            continue
        sr = load_condition(m, "swinir2_real2_1.0")
        bc = load_condition(m, "real2_1.0")
        sg, bg = sr.groupby("patch_id"), bc.groupby("patch_id")
        for pid, a in ref.groupby("patch_id"):
            xy = a[["x", "y"]].to_numpy()
            hit = {}
            for name, g in (("sr", sg), ("bc", bg)):
                b = g.get_group(pid) if pid in g.groups else None
                hit[name] = set(NM.mutual_nn(xy, b[["x", "y"]].to_numpy(), 12).i_a) if b is not None and len(b) else set()
            for i, (x, y) in enumerate(xy):
                s, c = i in hit["sr"], i in hit["bc"]
                cat = "both" if s and c else "sr_only" if s else "bic_only" if c else "missed"
                rows.append({"sample": a["sample"].iloc[0], "patch_id": pid, "x": x, "y": y, "cat": cat,
                             "ref_area_um2": a.area_um2.iloc[i]})
    d = pd.DataFrame(rows)
    return d[(d.x >= MARGIN) & (d.x < 512 - MARGIN) & (d.y >= MARGIN) & (d.y < 512 - MARGIN)]


def stratified(pool: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    pool = pool.assign(tert=pd.qcut(pool.ref_area_um2, 3, labels=False, duplicates="drop"))
    per = [n // 3 + (1 if k < n % 3 else 0) for k in range(3)]
    out = []
    for t, k in zip(range(3), per):
        g = pool[pool.tert == t].sample(frac=1, random_state=int(rng.integers(1 << 30)))
        g = pd.concat([x.assign(rank=np.arange(len(x))) for _, x in g.groupby("sample")]).sort_values(["rank", "sample"])
        out.append(g.head(k))
    return pd.concat(out)


def crop(sample: str, pid: str, x: float, y: float) -> tuple[np.ndarray, int, int]:
    img = cv2.imread(str(BENCH / "patches" / sample / "native" / f"{pid}.png"))
    x0, y0 = int(round(x)) - CROP // 2, int(round(y)) - CROP // 2
    return img[y0:y0 + CROP, x0:x0 + CROP], x0, y0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = ref_categories()
    picks = [stratified(d[d.cat == c], N[c], rng).assign(cat=c) for c in ("both", "sr_only", "bic_only", "missed")]
    r1 = pd.read_csv(ROOT / "results/sr08/annotation/labels_unblinded.csv")
    u = r1[(r1.cat == "fp_high") & (r1.label == "nucleus")]
    u = u[(u.x >= MARGIN) & (u.x < 512 - MARGIN) & (u.y >= MARGIN) & (u.y < 512 - MARGIN)]
    picks.append(u.sample(min(N["sr_unique"], len(u)), random_state=SEED).assign(cat="sr_unique", ref_area_um2=np.nan)
                 [["sample", "patch_id", "x", "y", "cat", "ref_area_um2"]])
    p = pd.concat(picks, ignore_index=True)
    order = rng.permutation(len(p))
    items, key = [], []
    for k, i in enumerate(order):
        r = p.iloc[i]
        c, x0, y0 = crop(r["sample"], r.patch_id, r.x, r.y)
        ok, buf = cv2.imencode(".png", c)
        iid = f"c{k + 1:03d}"
        items.append({"id": iid, "img": "data:image/png;base64," + base64.b64encode(buf).decode()})
        key.append({"id": iid, "cat": r["cat"], "sample": r["sample"], "patch_id": r.patch_id, "cx": r.x, "cy": r.y,
                    "crop_x0": x0, "crop_y0": y0, "ref_area_um2": r.ref_area_um2})
    (OUT / "items.json").write_text(json.dumps(items))
    k = pd.DataFrame(key)
    k.to_csv(OUT / "key.csv", index=False)
    print(len(k), k.cat.value_counts().to_dict(), "| samples", k["sample"].nunique(),
          f"| {(OUT / 'items.json').stat().st_size / 1e6:.1f} MB")
    print(k.groupby("cat").ref_area_um2.describe()[["min", "50%", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
