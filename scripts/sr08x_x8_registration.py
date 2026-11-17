"""Report fixed-frame Xenium to H&E centroid agreement without shifting evaluation coordinates."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from sr02_degrade_benchmark import find_file
from hemorph.nucleus_match import residual_shift

def main():
    patches = pd.read_csv(ROOT / 'results/sr_bench/xenium/patches_xenium.csv')
    rows = []
    for sid, group in patches.groupby('sample'):
        paths = [find_file(f'{kind}/{sid}_{name}.parquet') for kind, name in
                 [('xenium_seg', 'xenium_nucleus_seg'), ('cellvit_seg', 'cellvit_seg')]]
        xy = []
        for path in paths:
            frame = pd.read_parquet(path)
            xy.append(shapely.get_coordinates(shapely.centroid(shapely.from_wkb(frame.geometry.values))))
        mpp = float(group.src_mpp.iloc[0])
        a, b = [v * mpp for v in xy]
        d, _ = cKDTree(b).query(a)
        shift, retention = residual_shift(a, b, 20.0)
        rows.append(dict(sample=sid, xenium_path=str(paths[0]), cellvit_path=str(paths[1]),
                         source_mpp=mpp, n_xenium=len(a), n_cellvit=len(b),
                         median_nearest_um=float(np.median(d)), residual_dx_um=shift[0],
                         residual_dy_um=shift[1], residual_norm_um=float(np.linalg.norm(shift)),
                         fraction_within_20um=retention))
    out = ROOT / 'results/sr08/eval_x8'
    out.mkdir(parents=True, exist_ok=True)
    dest = out / 'registration_by_sample.csv'
    if dest.exists():
        raise FileExistsError(dest)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__ == '__main__':
    main()
