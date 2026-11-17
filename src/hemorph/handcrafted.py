"""Interpretable nuclear features (the 'H' baseline): shape from CellViT++ contours, stain from H&E.

All lengths in µm. Image = prepared model-resolution RGB (model px, same frame as contours).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .he_image import hed

SHAPE_COLS = ["area_um2", "perimeter_um", "circularity", "solidity", "eccentricity",
              "major_um", "minor_um", "aspect_ratio"]
STAIN_COLS = ["h_mean", "h_std", "h_p90", "h_cv", "h_lapvar", "e_ring_mean"]


def shape_features(contours: list[np.ndarray], mpp: float) -> pd.DataFrame:
    rows = []
    for c in contours:
        c = c.reshape(-1, 1, 2).astype(np.float32)
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        hull = cv2.contourArea(cv2.convexHull(c))
        if len(c) >= 5:
            (_, _), (d1, d2), _ = cv2.fitEllipse(c)
            major, minor = max(d1, d2), min(d1, d2)
        else:
            major = minor = np.sqrt(4 * a / np.pi)
        ecc = np.sqrt(1 - (minor / major) ** 2) if major > 0 else 0.0
        rows.append([a * mpp**2, p * mpp, 4 * np.pi * a / p**2 if p > 0 else np.nan,
                     a / hull if hull > 0 else np.nan, ecc, major * mpp, minor * mpp,
                     major / minor if minor > 0 else np.nan])
    return pd.DataFrame(rows, columns=SHAPE_COLS)


def stain_features(img: np.ndarray, contours: list[np.ndarray], ring_px: int = 6, tile: int = 2048) -> pd.DataFrame:
    """Hematoxylin intensity/texture inside the nucleus and eosin in a thin ring around it.

    Processed tile by tile; nuclei are assigned to the tile containing their bbox origin, and the
    tile is padded so contours crossing its edge are complete.
    """
    out = np.full((len(contours), len(STAIN_COLS)), np.nan, np.float32)
    boxes = np.array([[*c.min(0), *c.max(0)] for c in contours]) if contours else np.zeros((0, 4))
    pad = 64 + ring_px
    H, W = img.shape[:2]
    tid = (boxes[:, 1] // tile).astype(int) * 100_000 + (boxes[:, 0] // tile).astype(int)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    for t in np.unique(tid):
        ty, tx = divmod(t, 100_000)
        y0, x0 = max(ty * tile - pad, 0), max(tx * tile - pad, 0)
        y1, x1 = min((ty + 1) * tile + pad, H), min((tx + 1) * tile + pad, W)
        od = hed(img[y0:y1, x0:x1])
        h_ch, e_ch = od[..., 0], od[..., 1]
        lap = cv2.Laplacian(h_ch, cv2.CV_32F)
        for i in np.where(tid == t)[0]:
            bx0, by0, bx1, by1 = boxes[i].astype(int)
            bx0, by0 = max(bx0 - ring_px - x0, 0), max(by0 - ring_px - y0, 0)
            bx1, by1 = min(bx1 + ring_px + 1 - x0, x1 - x0), min(by1 + ring_px + 1 - y0, y1 - y0)
            if bx1 <= bx0 or by1 <= by0:
                continue
            mask = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
            cv2.fillPoly(mask, [np.round(contours[i] - [x0 + bx0, y0 + by0]).astype(np.int32)], 1)
            inside = mask.astype(bool)
            if inside.sum() < 5:
                continue
            ring = cv2.dilate(mask, kernel).astype(bool) & ~inside
            hv = h_ch[by0:by1, bx0:bx1][inside]
            out[i] = [hv.mean(), hv.std(), np.percentile(hv, 90), hv.std() / max(hv.mean(), 1e-6),
                      lap[by0:by1, bx0:bx1][inside].var(),
                      e_ch[by0:by1, bx0:bx1][ring].mean() if ring.any() else np.nan]
    return pd.DataFrame(out, columns=STAIN_COLS)
