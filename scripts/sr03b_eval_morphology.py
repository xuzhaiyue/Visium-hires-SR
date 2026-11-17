"""Step 03b — resolution vs morphology fidelity: CellViT++ results of each condition against the native reference.

Reference: CellViT++ SAM-H nuclei on the native mosaic (native resolution → 0.25 µm).
Each condition (gt = 0.5 µm, clean_/real_<mpp>, all upsampled to 0.25 µm with <method>) is compared with the reference within the same patch:
  detection   mutual nearest centroids within 3 µm count as matches → precision / recall / F1 (counts pooled per sample)
  nucleus     |log area ratio|, circularity and eccentricity differences of matched pairs (medians); type agreement
  patch       total-variation distance of cell-type composition; Spearman of patch mean nuclear area, density and inflammatory fraction with the reference
              (downstream morphology–expression analyses use patch/spot-level features, so this level is closest to actual use)
Outputs: results/sr_bench/eval/<method>/{metrics_by_sample.csv, metrics_by_group.csv, curves.png}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hemorph import cellvit_io as CV  # noqa: E402
from hemorph import handcrafted as HC  # noqa: E402
from hemorph import nucleus_match as NM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "results" / "sr_bench"
MODEL_MPP = 0.25
CELL = 512
MATCH_UM = 3.0
# v1: first sr02 degradation (JPEG on the low-res image + global fading; sr03c/sr03d showed it does not match real hires); v2: revised degradation, plus 3.8 µm
LEVELS_BY_VERSION = {
    "v1": {"gt": 0.5, "clean_1.0": 1.0, "real_1.0": 1.0, "clean_2.0": 2.0, "real_2.0": 2.0,
           "clean_3.25": 3.25, "real_3.25": 3.25},
    "v2": {"gt": 0.5, "clean_1.0": 1.0, "real2_1.0": 1.0, "clean_2.0": 2.0, "real2_2.0": 2.0,
           "clean_3.25": 3.25, "real2_3.25": 3.25, "clean_3.8": 3.8, "real2_3.8": 3.8},
}
LEVELS = LEVELS_BY_VERSION["v1"]
TYPES = ["Neoplastic", "Inflammatory", "Connective", "Epithelial", "Dead"]
# palette: categorical slots 1/2; text uses the text colour, not series colours
COL = {"FFPE": "#2a78d6", "Frozen (pancreas)": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_condition(method: str, cond: str) -> pd.DataFrame:
    d = BENCH / "mosaic" / f"{method}_{cond}"
    cells, contours = CV.load_cells(d / f"cellvit_{method}_{MODEL_MPP:g}mpp")
    lay = pd.read_csv(d / "layout.csv")
    pitch = CELL + (lay.ox.iloc[0])          # first ox column = GAP, pitch = CELL + GAP
    ncol = int(np.ceil(np.sqrt(len(lay))))
    col = ((cells.cx - lay.ox.iloc[0]) // pitch).astype(int)
    row = ((cells.cy - lay.oy.iloc[0]) // pitch).astype(int)
    idx = (row * ncol + col).to_numpy()
    lx = cells.cx.to_numpy() - lay.ox.iloc[0] - col.to_numpy() * pitch
    ly = cells.cy.to_numpy() - lay.oy.iloc[0] - row.to_numpy() * pitch
    ok = (lx >= 0) & (lx < CELL) & (ly >= 0) & (ly < CELL) & (idx >= 0) & (idx < len(lay))
    shp = HC.shape_features([contours[i] for i in np.flatnonzero(ok)], MODEL_MPP)
    out = pd.DataFrame({"patch_id": lay.patch_id.to_numpy()[idx[ok]], "sample": lay["sample"].to_numpy()[idx[ok]],
                        "x": lx[ok], "y": ly[ok], "cell_type": cells.cell_type.to_numpy()[ok]})
    return pd.concat([out, shp[["area_um2", "circularity", "eccentricity"]]], axis=1)


def patch_summary(df: pd.DataFrame, patch_ids) -> pd.DataFrame:
    g = df.groupby("patch_id")
    s = pd.DataFrame({"n": g.size(), "mean_area": g.area_um2.mean(),
                      "frac_infl": g.cell_type.apply(lambda t: (t == "Inflammatory").mean())})
    comp = pd.crosstab(df.patch_id, df.cell_type, normalize="index").reindex(columns=TYPES, fill_value=0)
    return s.join(comp).reindex(patch_ids)


def evaluate(ref: pd.DataFrame, pred: pd.DataFrame, patches: pd.DataFrame) -> pd.DataFrame:
    max_d = MATCH_UM / MODEL_MPP
    rows = []
    rp, pp = ref.groupby("patch_id"), pred.groupby("patch_id")
    sref, spred = patch_summary(ref, patches.patch_id), patch_summary(pred, patches.patch_id)
    for sid, ps in patches.groupby("sample"):
        tp = n_ref = n_pred = 0
        pairs = []
        for pid in ps.patch_id:
            a = rp.get_group(pid) if pid in rp.groups else ref.iloc[:0]
            b = pp.get_group(pid) if pid in pp.groups else pred.iloc[:0]
            n_ref += len(a)
            n_pred += len(b)
            if len(a) and len(b):
                m = NM.mutual_nn(a[["x", "y"]].to_numpy(), b[["x", "y"]].to_numpy(), max_d)
                tp += len(m)
                pairs.append(pd.DataFrame({
                    "log_area": np.log(b.area_um2.to_numpy()[m.i_b] / a.area_um2.to_numpy()[m.i_a]),
                    "d_circ": b.circularity.to_numpy()[m.i_b] - a.circularity.to_numpy()[m.i_a],
                    "d_ecc": b.eccentricity.to_numpy()[m.i_b] - a.eccentricity.to_numpy()[m.i_a],
                    "same_type": b.cell_type.to_numpy()[m.i_b] == a.cell_type.to_numpy()[m.i_a]}))
        pr = pd.concat(pairs) if pairs else pd.DataFrame(columns=["log_area", "d_circ", "d_ecc", "same_type"])
        p, r = tp / max(n_pred, 1), tp / max(n_ref, 1)
        a, b = sref.loc[ps.patch_id], spred.loc[ps.patch_id]
        tv = 0.5 * (a[TYPES].fillna(0) - b[TYPES].fillna(0)).abs().sum(axis=1)

        def rho(col):
            x, y = a[col], b[col]
            ok = x.notna() & y.notna()
            return spearmanr(x[ok], y[ok]).statistic if ok.sum() > 10 else np.nan

        rows.append({"sample": sid, "n_ref": n_ref, "n_pred": n_pred, "count_ratio": n_pred / max(n_ref, 1),
                     "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9),
                     "abs_log_area_med": pr.log_area.abs().median(), "log_area_bias": pr.log_area.median(),
                     "abs_d_circ_med": pr.d_circ.abs().median(), "abs_d_ecc_med": pr.d_ecc.abs().median(),
                     "type_agree": pr.same_type.mean(), "comp_tv_med": tv.median(),
                     "rho_density": rho("n"), "rho_mean_area": rho("mean_area"), "rho_frac_infl": rho("frac_infl")})
    return pd.DataFrame(rows)


PANELS = [("f1", "Nucleus detection F1", (0, 1)), ("abs_log_area_med", "Median |log area ratio|", None),
          ("comp_tv_med", "Cell-type composition error (TV)", None), ("rho_density", "Nuclear density Spearman ρ", (0, 1)),
          ("rho_mean_area", "Mean nuclear area Spearman ρ", (0, 1)), ("rho_frac_infl", "Inflammatory fraction Spearman ρ", (0, 1))]


def plot(grp: pd.DataFrame, dst: Path, method: str, real_kind: str = "real"):
    plt.rcParams.update({"font.family": ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"], "font.size": 9,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), constrained_layout=True)
    for ax, (m, title, ylim) in zip(axes.ravel(), PANELS):
        for g, c in COL.items():
            for kind, ls, mk in (("clean", "-", "o"), (real_kind, "--", "s")):
                s = grp[(grp.group == g) & (grp.kind.isin([kind, "gt"]))].sort_values("mpp")
                ax.plot(s.mpp, s[m], ls=ls, marker=mk, ms=6, lw=2, color=c, mfc=c if kind == "clean" else "white",
                        mew=1.5, label=f"{g} · {'downsampling only' if kind == 'clean' else 'degradation v1' if kind == 'real' else 'degradation v2'}")
        ax.set_xscale("log")
        ticks = sorted(grp.mpp.unique())
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        for lab in ax.get_xticklabels():          # 3.25 and 3.8 are close on the log axis; offset them
            lab.set_ha({"3.25": "right", "3.8": "left"}.get(lab.get_text(), "center"))
        ax.minorticks_off()
        ax.set_title(title, color=INK, loc="left", fontsize=10)
        ax.set_xlabel("Image pixel size (µm/px)")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if ylim:
            ax.set_ylim(*ylim)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=4, frameon=False, labelcolor=INK)
    fig.suptitle(f"Resolution vs morphology fidelity ({method} upsampling; reference = native CellViT++)", color=INK, fontsize=11)
    fig.savefig(dst, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="bicubic")
    ap.add_argument("--version", choices=list(LEVELS_BY_VERSION), default="v1")
    args = ap.parse_args()
    levels = LEVELS_BY_VERSION[args.version]
    out = BENCH / "eval" / (args.method if args.version == "v1" else f"{args.method}_{args.version}")
    out.mkdir(parents=True, exist_ok=True)
    patches = pd.read_csv(BENCH / "patches.csv")
    pilot = pd.read_csv(ROOT / "results/sr_survey/hest_pilot_samples.csv").set_index("id")
    grp_of = {s: ("FFPE" if pilot.loc[s, "pres"] == "FFPE" else "Frozen (pancreas)") for s in pilot.index}

    ref = load_condition(args.method, "native")
    res = []
    for cond, mpp in levels.items():
        d = BENCH / "mosaic" / f"{args.method}_{cond}"
        if not any(d.rglob("cells.json")):
            print(f"skip {cond}: no CellViT result yet")
            continue
        r = evaluate(ref, load_condition(args.method, cond), patches)
        r.insert(1, "condition", cond)
        r.insert(2, "kind", cond.split("_")[0])
        r.insert(3, "mpp", mpp)
        res.append(r)
    res = pd.concat(res)
    res.insert(1, "group", res["sample"].map(grp_of))
    res.to_csv(out / "metrics_by_sample.csv", index=False)
    num = res.select_dtypes("number").columns.drop("mpp")
    grp = res.groupby(["group", "condition", "kind", "mpp"])[list(num)].mean().reset_index()
    grp.to_csv(out / "metrics_by_group.csv", index=False)
    print(grp[["group", "condition", "f1", "recall", "precision", "abs_log_area_med", "comp_tv_med",
               "rho_density", "rho_mean_area", "rho_frac_infl"]].round(3).to_string(index=False))
    plot(grp, out / "curves.png", args.method, "real" if args.version == "v1" else "real2")
    print("done:", out)


if __name__ == "__main__":
    main()
