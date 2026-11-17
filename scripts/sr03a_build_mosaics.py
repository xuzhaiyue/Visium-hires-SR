"""Step 03a — tile the patches of each condition into one 0.25 µm/px mosaic WSI for a single CellViT++ run.

Conditions: native (native resolution → 0.25 µm, reference), gt (0.5 µm) and the sr02 clean_/real_<mpp> images; the latter are all upsampled to 0.25 µm with bicubic (CellViT++ x40 input scale),
i.e. this step produces the "bicubic baseline". SwinIR and other methods only replace the upsampling step; the mosaic layout is unchanged.
Patches are separated by 128 px (32 µm) white borders so that no CellViT++ nucleus spans two patches; the layout is written to layout.csv.
Lossless deflate compression avoids adding another layer of JPEG artefacts to every condition.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hemorph import he_image as HE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "results" / "sr_bench"
MODEL_MPP = 0.25
CELL = 512                     # 128 µm @ 0.25 µm
GAP = 128
CONDITIONS = ["native", "gt"] + [f"{k}_{m}" for m in (1.0, 2.0, 3.25) for k in ("clean", "real")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--method", default="bicubic", help="upsampling method label; sets the output subdirectory name")
    ap.add_argument("--patches", default=str(BENCH / "patches.csv"), help="only these patches (default: all)")
    args = ap.parse_args()
    patches = pd.read_csv(args.patches)
    ncol = int(np.ceil(np.sqrt(len(patches))))
    pitch = CELL + GAP
    layout = patches[["sample", "patch_id"]].copy()
    layout["ox"] = GAP + (np.arange(len(patches)) % ncol) * pitch
    layout["oy"] = GAP + (np.arange(len(patches)) // ncol) * pitch
    H = GAP + int(np.ceil(len(patches) / ncol)) * pitch
    W = GAP + ncol * pitch
    for cond in args.conditions:
        out = BENCH / "mosaic" / f"{args.method}_{cond}"
        out.mkdir(parents=True, exist_ok=True)
        mos = np.full((H, W, 3), 255, np.uint8)
        for r in layout.itertuples():
            img = cv2.imread(str(BENCH / "patches" / r.sample / cond / f"{r.patch_id}.png"))[..., ::-1]
            mos[r.oy:r.oy + CELL, r.ox:r.ox + CELL] = cv2.resize(img, (CELL, CELL), interpolation=cv2.INTER_CUBIC)
        dst = out / f"he_{args.method}_{MODEL_MPP:g}mpp.tiff"
        HE.write_pyramidal_tiff(mos, dst, MODEL_MPP, compression="zlib")
        layout.to_csv(out / "layout.csv", index=False)
        print(f"{cond}: {W}x{H} -> {dst}")


if __name__ == "__main__":
    main()
