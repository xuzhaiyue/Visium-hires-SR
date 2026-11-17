"""Archive the final replacement and analyze the complete 86-contour bundle separately."""
from pathlib import Path
import sys,json,shutil,hashlib
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.validation import explain_validity
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import sr08h_contour_eval as H
BASE=ROOT/'results/sr08/contours_x4_supplement_2026-10-23'

def main():
    src=Path(sys.argv[1]);rep=BASE/'replacement01'
    dest=rep/'raw'/src.name
    assert not dest.exists();shutil.copyfile(src,dest)
    raw=json.loads(dest.read_text());assert set(raw['records'])=={'x4r001'}
    row=raw['records']['x4r001'];assert row['status']=='drawn'
    a=np.asarray(row['pts']);assert a.shape[0]>=3 and ((a>=0)&(a<=128)).all()
    poly=Polygon(a);fixed=poly.buffer(0);assert not fixed.is_empty and fixed.geom_type=='Polygon'
    out=BASE/'final_analysis_2026-10-23';out.mkdir(exist_ok=False)
    qc=dict(id='x4r001',n_points=len(a),raw_polygon_valid=poly.is_valid,validity=explain_validity(poly),area_change_relative=abs(fixed.area-poly.area)/poly.area,raw_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (out/'replacement_qc.json').write_text(json.dumps(qc,indent=2))
    bundle=out/'derived_bundle';shutil.copytree(BASE/'combined_analysis_2026-10-23/derived_bundle',bundle)
    (bundle/'raw/contours/x4r001.json').write_text(json.dumps(row))
    key=pd.read_csv(bundle/'key.csv');add=pd.read_csv(rep/'key.csv').rename(columns={'cat_x4':'cat'})
    pd.concat([key,add],ignore_index=True).to_csv(bundle/'key.csv',index=False)
    H.C=bundle;H.OUT=out/'eval';H.CONDS={'native (upper bound)':'native','bicubic ×4':'real2_2.0','SwinIR ×4':'swinir4_real2_2.0'}
    H.main()
    coverage=pd.read_csv(BASE/'combined_analysis_2026-10-23/updated_coverage.csv')
    mask=coverage['cat'].eq('bic_only')&coverage.area_tertile.eq(3)
    coverage.loc[mask,'combined_drawn_n']+=1
    coverage['remaining_gap']=(coverage.proposed_target-coverage.combined_drawn_n).clip(lower=0)
    assert coverage.remaining_gap.sum()==0
    coverage.to_csv(out/'final_coverage.csv',index=False)
    metrics=pd.read_csv(H.OUT/'per_nucleus.csv');assert metrics.id.nunique()==86
    strict=metrics[~metrics.id.isin(pd.read_csv(BASE/'qc_2026-10-23/trajectory_qc.csv').query('valid == False').id)]
    if not poly.is_valid:strict=strict[strict.id.ne('x4r001')]
    strict.to_csv(out/'originally_valid_contours_sensitivity.csv',index=False)
    (out/'run_config.yaml').write_text(json.dumps(dict(run_utc=datetime.now(timezone.utc).isoformat(),raw=str(dest),raw_sha256=qc['raw_sha256'],n_old=57,n_supplement=28,n_replacement=1,n_total=86,unable_retained='x4c001',frozen_geometry_processing='sr08h unchanged',sampling_cell_minimum=5,coverage_remaining_gap=0),indent=2))
    (out/'README.md').write_text('''# ×4 supplementary contours complete

57 original + 28 supplementary + 1 replacement = 86 traced nuclei. Untraceable item x4c001 is kept but excluded; x4r001 is the first pre-reserved candidate from the same category/area tertile, not chosen by appearance. Every native category × area tertile reaches the approved minimum of 5 traced nuclei.
Raw JSON and the earlier 57- and 85-nucleus analyses are not overwritten; all metrics use the sr08h nearest-nucleus matching and smoothing/buffer(0) geometry with no new thresholds. Per-nucleus data excluding self-intersecting raw traces are saved separately as a sensitivity analysis.
Paired comparisons on commonly detected nuclei are in eval/summary.txt; marginal medians over different detection sets are not evidence of paired improvement.
''')
    print('Final contour coverage: all 12 cells covered; 86 drawn nuclei')
if __name__=='__main__':main()
