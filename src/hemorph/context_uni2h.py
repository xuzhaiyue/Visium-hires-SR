"""Neighbourhood ('B') features from UNI2-h with the centre region hidden.

mode='ring'          : 8 native 112 µm tiles arranged around the region, none overlapping it;
                       B = mean embedding (+ optional concat). The model never sees the centre.
mode='masked_center' : one larger window (2·(half + 112) µm) resized to 224 px, centre filled
                       with the slide's background-free median colour. Sensitivity only — the
                       resize changes magnification and the grey square is out of distribution.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

UNI2H_KWARGS = dict(  # from the MahmoodLab/UNI2-h model card — check against your existing UNI2-h pipeline
    img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
    mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, reg_tokens=8, dynamic_img_size=True)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def load_uni2h(weights: str | None, device: str):
    import timm
    import torch
    kw = dict(UNI2H_KWARGS, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
    if weights:
        model = timm.create_model("vit_giant_patch14_224", pretrained=False, **kw)
        model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    else:
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw)
    return model.eval().to(device)


def ring_centers(cx: float, cy: float, half_um: float, tile_um: float, mpp: float) -> np.ndarray:
    d = (half_um + tile_um / 2) / mpp
    offs = np.array([(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)], float)
    return np.array([cx, cy]) + offs * d


def crop(img: np.ndarray, cx: float, cy: float, size_px: int, fill: np.ndarray) -> np.ndarray:
    """Square crop centred at (cx, cy); out-of-image area filled with `fill`."""
    H, W = img.shape[:2]
    x0, y0 = int(round(cx - size_px / 2)), int(round(cy - size_px / 2))
    out = np.empty((size_px, size_px, 3), np.uint8)
    out[:] = fill
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + size_px, W), min(y0 + size_px, H)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def tissue_fraction(tile: np.ndarray) -> float:
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    return float((hsv[..., 1] > 20).mean())


@np.errstate(invalid="ignore")
def embed(model, tiles: list[np.ndarray], device: str, batch: int = 64) -> np.ndarray:
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(tiles), batch):
            x = np.stack(tiles[i:i + batch]).astype(np.float32) / 255.0
            x = (x - MEAN) / STD
            x = torch.from_numpy(x.transpose(0, 3, 1, 2)).float().to(device)
            out.append(model(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, UNI2H_KWARGS["embed_dim"]), np.float32)


def region_context(model, img: np.ndarray, centers: np.ndarray, half_um: float, mpp: float, device: str,
                   mode: str = "ring", tile_um: float = 112.0, min_tissue: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Returns (B features [n_regions x D], tissue fraction of the context [n_regions])."""
    fill = np.median(img[::64, ::64].reshape(-1, 3), axis=0).astype(np.uint8)
    tile_px = int(round(tile_um / mpp))
    D = UNI2H_KWARGS["embed_dim"]
    if mode == "ring":
        # neighbouring regions share ring tiles -> embed each unique tile centre once
        rc = np.stack([ring_centers(cx, cy, half_um, tile_um, mpp) for cx, cy in centers])  # n x 8 x 2
        uniq, inv = np.unique(np.round(rc.reshape(-1, 2), 1), axis=0, return_inverse=True)
        tiles = [crop(img, x, y, tile_px, fill) for x, y in uniq]
        tiles = [cv2.resize(t, (224, 224), interpolation=cv2.INTER_AREA) if t.shape[0] != 224 else t for t in tiles]
        tf = np.array([tissue_fraction(t) for t in tiles])
        keep = tf >= min_tissue
        E = np.full((len(uniq), D), np.nan, np.float32)
        E[keep] = embed(model, [t for t, k in zip(tiles, keep) if k], device)
        E, tf = E[inv.ravel()].reshape(len(centers), 8, D), tf[inv.ravel()].reshape(len(centers), 8)
        with np.errstate(invalid="ignore"):
            return np.nanmean(E, axis=1), tf.mean(1)
    feats, tfrac = [], []
    for cx, cy in centers:
        if mode == "masked_center":
            win_px = int(round(2 * (half_um + tile_um) / mpp))
            t = crop(img, cx, cy, win_px, fill)
            h = int(round(half_um / mpp))
            c = win_px // 2
            t[c - h:c + h, c - h:c + h] = fill
            t = cv2.resize(t, (224, 224), interpolation=cv2.INTER_AREA)
            feats.append(embed(model, [t], device)[0])
            tfrac.append(tissue_fraction(t))
        else:
            raise ValueError(mode)
    return np.vstack(feats), np.array(tfrac)


def save(path: Path, B: np.ndarray, tfrac: np.ndarray, keys: np.ndarray):
    np.savez_compressed(path, B=B, tissue_frac=tfrac, region_key=keys)
