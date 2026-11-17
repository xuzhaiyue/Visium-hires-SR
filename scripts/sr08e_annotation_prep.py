"""Step 08e — expert labelling of ×2 false positives: prepare blinded items.

Question: are nuclei detected by SwinIR but absent from the native CellViT reference (automatic FPs) real nuclei missed by the reference, or hallucinations?
All items come from the HEST native image (native resolution → 0.25 µm); each item is a 64 µm crop (256 px) centred on the candidate, asking only
"is there a nucleus at the circle in the native image". Three kinds of items are mixed and shuffled; the page shows only an ID and the image; the category key is stored locally in key.csv:
  fp_low   150: detected by SwinIR, not by the reference, from samples with native 0.34–0.55 µm (stratum with most FPs)
  fp_high   50: as above, from samples with native ≤ 0.3 µm
  tp        50: detected by both reference and SwinIR ("nucleus" consistency control)
  bg        50: random tissue points ≥ 8 µm from any reference / SwinIR nucleus ("not a nucleus" control)
Each category is sampled evenly across samples with a fixed seed. Outputs: results/sr08/annotation/{items.json, key.csv}
"""
from __future__ import annotations

import base64
import json
import zlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from hemorph import nucleus_match as NM  # noqa: E402
from sr03b_eval_morphology import load_condition  # noqa: E402

BENCH = ROOT / "results" / "sr_bench"
OUT = ROOT / "results" / "sr08" / "annotation"
CROP = 256            # 64 µm @ 0.25 µm
EDGE = 16             # skip candidates < 4 µm from the patch edge
SEED = 7
N = {"fp_low": 150, "fp_high": 50, "tp": 50, "bg": 50}


def collect() -> pd.DataFrame:
    meta = pd.read_csv(ROOT / "results/sr_survey/hest_benchmark_candidates.csv").set_index("id")
    rows = []
    for m in ("bicubic", "bicubic2"):
        ref = load_condition(m, "native")
        sr = load_condition(m, "swinir2_real2_1.0")
        rg, sg = ref.groupby("patch_id"), sr.groupby("patch_id")
        for pid, b in sg:
            a = rg.get_group(pid) if pid in rg.groups else ref.iloc[:0]
            sid = b["sample"].iloc[0]
            mt = NM.mutual_nn(a[["x", "y"]].to_numpy(), b[["x", "y"]].to_numpy(), 12) if len(a) else pd.DataFrame(
                {"i_a": [], "i_b": []})
            matched_b = set(mt.i_b.astype(int))
            strat = "fp_high" if meta.loc[sid, "px"] <= 0.3 else "fp_low"
            for j, (x, y) in enumerate(b[["x", "y"]].to_numpy()):
                rows.append({"sample": sid, "patch_id": pid, "x": x, "y": y,
                             "cat": "tp" if j in matched_b else strat})
            # background points: inside tissue, ≥ 8 µm from all nuclei
            pts = np.vstack([a[["x", "y"]].to_numpy(), b[["x", "y"]].to_numpy()]) if len(a) else b[["x", "y"]].to_numpy()
            tree = cKDTree(pts)
            rng = np.random.default_rng([SEED, zlib.crc32(pid.encode())])
            for _ in range(3):
                x, y = rng.uniform(EDGE, 512 - EDGE, 2)
                if tree.query([x, y])[0] >= 32:
                    rows.append({"sample": sid, "patch_id": pid, "x": x, "y": y, "cat": "bg_cand"})
    d = pd.DataFrame(rows)
    return d[(d.x >= EDGE) & (d.x < 512 - EDGE) & (d.y >= EDGE) & (d.y < 512 - EDGE)]


def crop(sample: str, pid: str, x: float, y: float) -> np.ndarray:
    img = cv2.imread(str(BENCH / "patches" / sample / "native" / f"{pid}.png"))
    pad = cv2.copyMakeBorder(img, CROP, CROP, CROP, CROP, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cx, cy = int(round(x)) + CROP, int(round(y)) + CROP
    return pad[cy - CROP // 2:cy + CROP // 2, cx - CROP // 2:cx + CROP // 2]


def tissue_ok(c: np.ndarray) -> bool:
    return (cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 1] > 20).mean() > 0.4


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = collect()
    rng = np.random.default_rng(SEED)
    picks = []
    for cat, n in N.items():
        pool = d[d.cat == ("bg_cand" if cat == "bg" else cat)].sample(frac=1, random_state=SEED)
        per = pool.groupby("sample")
        order = pd.concat([g.assign(rank=np.arange(len(g))) for _, g in per]).sort_values(["rank", "sample"])
        got = []
        for r in order.itertuples():                  # round-robin over samples for balance
            c = crop(r.sample, r.patch_id, r.x, r.y)
            if cat == "bg" and not tissue_ok(c):
                continue
            got.append((cat, r, c))
            if len(got) >= n:
                break
        picks += got
    order = rng.permutation(len(picks))
    items, key = [], []
    for k, i in enumerate(order):
        cat, r, c = picks[i]
        iid = f"q{k + 1:03d}"
        ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 90])
        items.append({"id": iid, "img": "data:image/jpeg;base64," + base64.b64encode(buf).decode()})
        key.append({"id": iid, "cat": cat, "sample": r.sample, "patch_id": r.patch_id, "x": r.x, "y": r.y})
    (OUT / "items.json").write_text(json.dumps(items))
    pd.DataFrame(key).to_csv(OUT / "key.csv", index=False)
    k = pd.DataFrame(key)
    print(k.cat.value_counts().to_dict(), "| samples per category", k.groupby("cat")["sample"].nunique().to_dict())
    print(f"items.json {(OUT / 'items.json').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
