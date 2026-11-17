"""Match CellViT++ nuclei (model px) to Space Ranger nuclei (full-res px).

The transform is known (scale = model_mpp / src_mpp, shared origin); matching verifies it and
estimates any residual shift. A low match rate means the wrong H&E file or a registration problem —
stop and fix before modelling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def to_fullres(xy_model: np.ndarray, model_mpp: float, src_mpp: float, shift=(0.0, 0.0)) -> np.ndarray:
    return xy_model * (model_mpp / src_mpp) + np.asarray(shift)


def residual_shift(xy_a: np.ndarray, xy_b: np.ndarray, max_d: float) -> tuple[np.ndarray, float]:
    """Median offset b - a among nearest neighbours within max_d; returns (shift, match_rate)."""
    d, j = cKDTree(xy_b).query(xy_a, distance_upper_bound=max_d)
    ok = np.isfinite(d)
    if ok.sum() < 100:
        return np.zeros(2), ok.mean()
    return np.median(xy_b[j[ok]] - xy_a[ok], axis=0), ok.mean()


def mutual_nn(xy_a: np.ndarray, xy_b: np.ndarray, max_d: float) -> pd.DataFrame:
    """Mutual nearest neighbours within max_d. Returns rows (i_a, i_b, dist)."""
    da, ja = cKDTree(xy_b).query(xy_a, distance_upper_bound=max_d)
    db, jb = cKDTree(xy_a).query(xy_b, distance_upper_bound=max_d)
    ia = np.where(np.isfinite(da))[0]
    mutual = jb[ja[ia]] == ia
    ia = ia[mutual]
    return pd.DataFrame({"i_a": ia, "i_b": ja[ia], "dist": da[ia]})


def match(cv: pd.DataFrame, tenx: pd.DataFrame, model_mpp: float, src_mpp: float, max_dist_um: float) -> tuple[pd.DataFrame, dict]:
    """cv: CellViT table (cx, cy in model px); tenx: 10x nuclei (x, y in full-res px, index cell_id)."""
    max_d = max_dist_um / src_mpp
    xy_cv = to_fullres(cv[["cx", "cy"]].to_numpy(), model_mpp, src_mpp)
    xy_tx = tenx[["x", "y"]].to_numpy()
    shift, rate0 = residual_shift(xy_cv, xy_tx, 3 * max_d)
    xy_cv = xy_cv + shift
    m = mutual_nn(xy_cv, xy_tx, max_d)
    out = pd.DataFrame(index=cv.index)
    out["x_full"], out["y_full"] = xy_cv[:, 0], xy_cv[:, 1]
    out["cell_id"] = pd.Series(tenx.index.to_numpy()[m.i_b.to_numpy()], index=cv.index[m.i_a.to_numpy()])
    out["match_dist_um"] = pd.Series(m.dist.to_numpy() * src_mpp, index=cv.index[m.i_a.to_numpy()])
    qc = {"residual_shift_um": (shift * src_mpp).round(2).tolist(),
          "frac_cellvit_matched": float(out.cell_id.notna().mean()),
          "frac_10x_matched": float(len(m) / len(tenx)),
          "median_dist_um": float(np.median(m.dist) * src_mpp) if len(m) else np.nan}
    return out, qc
