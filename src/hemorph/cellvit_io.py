"""Load CellViT++ inference outputs into a per-nucleus table + embedding matrix.

CellViT++ 1.0.9 (`--graph`) writes, per slide, in <outdir>/<slide stem>/:
  cells.json        {wsi_metadata, type_map, cells: [{centroid [x,y], contour [[x,y]...], bbox
                     [[row0,col0],[row1,col1]], type (1-based, see type_map), type_prob, ...}]}
  cells.pt          cellvit CellGraphDataWSI (x: N×384 tokens, positions, nuclei_types) —
                    unpickles only with `cellvit` installed, so scripts/export_cellvit_graph.py
                    turns it into cells_graph.npz (same row order as cells.json, checked).
Coordinates are level-0 pixels of the image given to CellViT++ (model px).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _slide_dir(outdir: str | Path) -> Path:
    hits = sorted(Path(outdir).rglob("cells.json"))
    if not hits:
        raise FileNotFoundError(f"no cells.json under {outdir}")
    return hits[0].parent


def load_cells(outdir: str | Path) -> tuple[pd.DataFrame, list[np.ndarray]]:
    d = _slide_dir(outdir)
    data = json.loads((d / "cells.json").read_text())
    type_map = {int(k): v for k, v in data["type_map"].items()}
    rows, contours = [], []
    for c in data["cells"]:
        cx, cy = c["centroid"]
        t = int(c["type"])
        pr, pc = c.get("patch_coordinates", (-1, -1))
        rows.append({"cx": float(cx), "cy": float(cy), "type_id": t, "cell_type": type_map.get(t, str(t)),
                     "type_prob": float(c.get("type_prob", np.nan)), "patch_row": int(pr), "patch_col": int(pc)})
        contours.append(np.asarray(c["contour"], dtype=np.float32))
    df = pd.DataFrame(rows)
    df.index.name = "nucleus_idx"
    print(f"[cellvit] {d}: {len(df)} nuclei; types={df.cell_type.value_counts().to_dict()}")
    return df, contours


def load_embeddings(outdir: str | Path, n_expected: int, centroids: np.ndarray | None = None) -> np.ndarray:
    """Token embeddings aligned to load_cells() order."""
    d = _slide_dir(outdir)
    npz = d / "cells_graph.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} missing — run scripts/export_cellvit_graph.py {d} in the cellvit env")
    g = np.load(npz)
    x = g["x"]
    if x.shape[0] != n_expected:
        raise ValueError(f"{npz.name}: {x.shape[0]} embeddings vs {n_expected} cells in json")
    if centroids is not None and np.abs(g["positions"] - centroids).max() > 1:
        raise ValueError("embedding positions disagree with json centroids")
    return x
