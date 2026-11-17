"""Step 08d — stage 4: independent nuclear-morphology evaluation (plan §4–§5).

  infer   upsample the low-res test patches (real2_<0.5·s>) to 0.5 µm with results/sr08/models/x<s>/best.pth, stored as swinir<s>_real2_<lr>;
          Xenium patches have no 1.0 / 2.0 / 4.0 µm levels; they are first generated from gt (0.5 µm) with the sr02 v2 degradation (fixed seed).
  mosaic  build 0.25 µm mosaics per batch (sr03a: SR output ×2 bicubic; bicubic baseline upsampled directly from the low-res image).
          CellViT++ (MODEL=SAM) is then run on these mosaics.
  eval    references:
            HEST     CellViT SAM-H nuclei on the native image (automatic reference labels)
            Xenium   Xenium DAPI nucleus segmentation (fully independent of H&E / CellViT); upper bound = CellViT on the 0.5 µm image
          per sample: detection P / R / F1, FDR = FP / (TP + FP), medians of |log area ratio| and circularity difference of matched nuclei;
          match = mutual nearest centroids within 3 µm. SwinIR vs bicubic paired Wilcoxon, FFPE / frozen separately.
          Pre-specified criteria per level: ΔF1 >= 0.05, relative area-error reduction >= 10%, circularity error not worse, ΔFDR <= 0.02.
Outputs: results/sr08/eval_x<s>/
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from sr02_degrade_benchmark import GT_MPP, degrade_real2, find_file, save_png  # noqa: E402
from sr03b_eval_morphology import evaluate, load_condition  # noqa: E402
from hemorph import handcrafted as HC  # noqa: E402

BENCH = ROOT / "results" / "sr_bench"
SR08 = ROOT / "results" / "sr08"
XEN = BENCH / "xenium"
MODEL_MPP = 0.25
PILOT = ["NCBI569", "NCBI570", "NCBI571", "NCBI572", "TENX179", "MISC38", "TENX39", "TENX46", "TENX62"]


def lr_tag(scale: int) -> str:
    return f"real2_{0.5 * scale:.1f}"


def test_sets() -> dict[str, pd.DataFrame]:
    hest = pd.read_csv(BENCH / "patches.csv")
    xen = pd.read_csv(XEN / "patches_xenium.csv")
    return {"bicubic": hest[hest["sample"].isin(PILOT)], "bicubic2": hest[~hest["sample"].isin(PILOT)],
            "xenium": xen}


# ---------- infer ----------

def make_xenium_lr(scale: int):
    xen = pd.read_csv(XEN / "patches_xenium.csv")
    n = int(round(256 * GT_MPP / (0.5 * scale)))
    for r in xen.itertuples():
        dst = BENCH / "patches" / r.sample / lr_tag(scale) / f"{r.patch_id}.png"
        if dst.exists():
            continue
        rng = np.random.default_rng([scale, int(r.patch_id.split("_x")[1]), int.from_bytes(r.sample.encode(), "little") % 2 ** 31])
        gt = cv2.imread(str(BENCH / "patches" / r.sample / "gt" / f"{r.patch_id}.png"))[..., ::-1]
        save_png(dst, degrade_real2(gt, n, rng)[0])


def infer(scale: int, batch: int = 4):
    import torch
    from sr08c_train_swinir import build
    torch.set_num_threads(2)
    make_xenium_lr(scale)
    model = build(scale, pretrained=False)
    model.load_state_dict(torch.load(SR08 / "models" / f"x{scale}" / "best.pth", map_location="cpu"))
    model = model.eval().to("mps")
    src, dst = lr_tag(scale), f"swinir{scale}_{lr_tag(scale)}"
    allp = pd.concat(test_sets().values())
    todo = [r for r in allp.itertuples() if not (BENCH / "patches" / r.sample / dst / f"{r.patch_id}.png").exists()]
    with torch.no_grad():
        for i in range(0, len(todo), batch):
            while os.getloadavg()[0] > 8:
                time.sleep(10)
            rs = todo[i:i + batch]
            x = np.stack([cv2.imread(str(BENCH / "patches" / r.sample / src / f"{r.patch_id}.png"))[..., ::-1] for r in rs])
            y = model(torch.from_numpy(x.transpose(0, 3, 1, 2).astype(np.float32) / 255).to("mps")).clamp(0, 1)
            y = (y.cpu().numpy().transpose(0, 2, 3, 1) * 255 + 0.5).astype(np.uint8)
            for r, im in zip(rs, y):
                save_png(BENCH / "patches" / r.sample / dst / f"{r.patch_id}.png", im)
            if i % 100 == 0:
                print(f"Inference {i + len(rs)}/{len(todo)}", flush=True)
            time.sleep(0.5)
    print(f"SR inference done: {len(todo)} patches → {dst}")


def mosaic(scale: int):
    sets = test_sets()
    for method, p in sets.items():
        f = SR08 / (f"patches_{method}_x{scale}.csv" if scale in (4, 8) else f"patches_{method}.csv")
        p.to_csv(f, index=False)
        conds = [f"swinir{scale}_{lr_tag(scale)}", lr_tag(scale)] + (["gt"] if method == "xenium" else [])
        conds = [c for c in conds if not any((BENCH / "mosaic" / f"{method}_{c}").rglob("cells.json"))]
        if conds:
            subprocess.run([sys.executable, str(ROOT / "scripts/sr03a_build_mosaics.py"), "--method", method,
                            "--patches", str(f), "--conditions", *conds], check=True)
            print(f"{method}: conditions needing CellViT {conds}")


# ---------- eval ----------

def xenium_reference(p: pd.DataFrame) -> pd.DataFrame:
    """Xenium DAPI nuclei → same format as load_condition (0.25 µm pixel coordinates within the patch + morphology)."""
    rows = []
    for sid, ps in p.groupby("sample", sort=False):
        seg = pd.read_parquet(find_file(f"xenium_seg/{sid}_xenium_nucleus_seg.parquet"))
        g = shapely.from_wkb(seg.geometry.values)
        c = shapely.get_coordinates(shapely.centroid(g))
        for r in ps.itertuples():
            k = np.flatnonzero((c[:, 0] >= r.x0) & (c[:, 0] < r.x0 + r.src_px) & (c[:, 1] >= r.y0) & (c[:, 1] < r.y0 + r.src_px))
            s = r.src_mpp / MODEL_MPP
            conts = [(shapely.get_coordinates(shapely.get_exterior_ring(g[i])) - [r.x0, r.y0]) * s for i in k]
            shp = HC.shape_features([cc.astype(np.float32) for cc in conts], MODEL_MPP)
            rows.append(pd.concat([pd.DataFrame({"patch_id": r.patch_id, "sample": sid,
                                                 "x": (c[k, 0] - r.x0) * s, "y": (c[k, 1] - r.y0) * s,
                                                 "cell_type": "nucleus"}),
                                   shp[["area_um2", "circularity", "eccentricity"]]], axis=1))
    return pd.concat(rows, ignore_index=True)


def run_eval(scale: int):
    from scipy.stats import wilcoxon
    out = SR08 / f"eval_x{scale}"
    out.mkdir(parents=True, exist_ok=True)
    sets = test_sets()
    hest_meta = pd.read_csv(ROOT / "results/sr_survey/hest_benchmark_candidates.csv").set_index("id")
    rows = []
    for method, p in sets.items():
        if method == "xenium":
            ref = xenium_reference(p)
            conds = {"upper bound: 0.5 µm CellViT": "gt", "bicubic": lr_tag(scale), "SwinIR": f"swinir{scale}_{lr_tag(scale)}"}
        else:
            ref = load_condition(method, "native")
            conds = {"bicubic": lr_tag(scale), "SwinIR": f"swinir{scale}_{lr_tag(scale)}"}
        for name, cond in conds.items():
            r = evaluate(ref, load_condition(method, cond), p)
            r.insert(1, "method", name)
            r.insert(2, "testset", "Xenium (DAPI nucleus reference)" if method == "xenium" else "HEST (native CellViT reference)")
            rows.append(r)
    res = pd.concat(rows, ignore_index=True)
    res["FDR"] = 1 - res.precision
    pres = {s: hest_meta.loc[s, "pres"] if s in hest_meta.index else "FFPE" for s in res["sample"]}   # all Xenium samples are FFPE
    res["group"] = res["sample"].map(lambda s: {"FFPE": "FFPE", "Fresh Frozen": "frozen"}.get(pres[s], "not stated"))
    res["native_mpp"] = res["sample"].map(hest_meta["px"])
    res["stratum"] = np.where(res["testset"].str.startswith("Xenium"), "Xenium auxiliary",
                              np.where(res["native_mpp"] <= 0.3, "primary", "generalisation"))
    res.to_csv(out / "metrics_by_sample.csv", index=False)

    cols = ["f1", "recall", "precision", "FDR", "abs_log_area_med", "abs_d_circ_med", "count_ratio"]
    pd.set_option("display.width", 220)
    print(res.groupby(["testset", "group", "method"])[cols].mean().round(3).to_string())

    lines = ["# sr08 ×%d independent evaluation: SwinIR vs bicubic (paired by sample)\n" % scale]
    grouping = "stratum" if scale in (4, 8) else "group"
    for (ts, grp), g in res.groupby(["testset", grouping]):
        w = g.pivot(index="sample", columns="method", values=cols)
        d = {c: (w[(c, "SwinIR")] - w[(c, "bicubic")]).dropna() for c in cols}
        dF1, dFDR = d["f1"].mean(), d["FDR"].mean()
        a_b, a_s = w[("abs_log_area_med", "bicubic")].mean(), w[("abs_log_area_med", "SwinIR")].mean()
        c_b, c_s = w[("abs_d_circ_med", "bicubic")].mean(), w[("abs_d_circ_med", "SwinIR")].mean()
        area_rel = (a_b - a_s) / a_b
        ok = {"ΔF1 ≥ 0.05": dF1 >= 0.05, "relative area-error reduction ≥ 10%": area_rel >= 0.10,
              "circularity error not worse": c_s <= c_b, ("automatic ΔFDR ≤ 0.02 (descriptive)" if scale in (4, 8) else "ΔFDR ≤ 0.02"): dFDR <= 0.02}
        p = lambda x: wilcoxon(x).pvalue if len(x) >= 5 and (x != 0).any() else np.nan
        lines += [f"\n## {ts} · {grp} ({len(w)} samples)",
                  f"- F1: bicubic {w[('f1', 'bicubic')].mean():.3f} → SwinIR {w[('f1', 'SwinIR')].mean():.3f}"
                  f" (Δ {dF1:+.3f}, {int((d['f1'] > 0).sum())}/{len(d['f1'])} improved, p = {p(d['f1']):.3g})",
                  f"- FDR: {w[('FDR', 'bicubic')].mean():.3f} → {w[('FDR', 'SwinIR')].mean():.3f} (Δ {dFDR:+.3f}, p = {p(d['FDR']):.3g})",
                  f"- |log area ratio|: {a_b:.3f} → {a_s:.3f} (relative change {-area_rel:+.1%}, p = {p(d['abs_log_area_med']):.3g})",
                  f"- circularity error: {c_b:.3f} → {c_s:.3f} (p = {p(d['abs_d_circ_med']):.3g})",
                  "- criteria: " + "; ".join(f"{k} {'✓' if v else '✗'}" for k, v in ok.items()),
                  ("- **overall decision pending: requires slide-bootstrap CI and conservative analysis of the expert-corrected FDR; automatic FDR is not used as a final criterion.**"
                   if scale in (4, 8) else f"- **{'pass' if all(ok.values()) else 'fail'}**")]
    (out / "VERDICT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["infer", "mosaic", "eval"])
    ap.add_argument("--scale", type=int, choices=[2, 4, 8], default=2)
    args = ap.parse_args()
    {"infer": infer, "mosaic": mosaic, "eval": run_eval}[args.step](args.scale)


if __name__ == "__main__":
    main()
