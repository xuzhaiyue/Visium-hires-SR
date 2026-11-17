"""Build an auditable submitted-copy bundle after the user confirms completed drawings."""
from pathlib import Path
import sys,json,shutil
from datetime import datetime,timezone
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import sr08h_contour_eval as H
BASE=ROOT/'results/sr08/contours_x4_supplement_2026-10-23'

def main():
    out=BASE/'combined_analysis_2026-10-23';out.mkdir(exist_ok=False)
    bundle=out/'derived_bundle';(bundle/'raw/contours').mkdir(parents=True)
    old=ROOT/'results/sr08/contours'
    for p in (old/'raw/contours').glob('*.json'):
        shutil.copyfile(p,bundle/'raw/contours'/p.name)
    keys=pd.read_csv(old/'key.csv')
    add=pd.read_csv(BASE/'key.csv').rename(columns={'cat_x4':'cat'})
    raw=BASE/'raw/sr08_x4_contours_1792785694475.json'
    records=json.loads(raw.read_text())['records']
    for iid,row in records.items():
        derived=dict(row)
        if row['status']=='draft':
            assert len(row.get('pts',[]))>=3
            derived['status']='drawn'
            derived['submission_evidence']='annotator confirmed tracing finished; original raw status remains draft'
        (bundle/'raw/contours'/f'{iid}.json').write_text(json.dumps(derived))
    pd.concat([keys,add],ignore_index=True).to_csv(bundle/'key.csv',index=False)
    H.C=bundle;H.OUT=out/'eval'
    H.CONDS={'native (upper bound)':'native','bicubic ×4':'real2_2.0','SwinIR ×4':'swinir4_real2_2.0'}
    H.main()
    mask=pd.DataFrame([dict(id=i,original_status=v['status'],analysis_status='drawn' if v['status']=='draft' else v['status'],has_contour=len(v.get('pts',[]))>=3) for i,v in records.items()])
    mask.to_csv(out/'submission_mask.csv',index=False)
    audit=ROOT/'results/sr08/contour_coverage_x4_2026-10-23'
    gaps=pd.read_csv(audit/'coverage_and_proposed_gaps.csv')
    additions=add[add.id.isin(mask[mask.analysis_status.eq('drawn')].id)].groupby(['cat','area_tertile']).size().rename('new_drawn_n').reset_index()
    coverage=gaps.merge(additions,on=['cat','area_tertile'],how='left').fillna({'new_drawn_n':0})
    coverage['combined_drawn_n']=coverage.drawn_n+coverage.new_drawn_n
    coverage['remaining_gap']=(coverage.proposed_target-coverage.combined_drawn_n).clip(lower=0)
    coverage.to_csv(out/'updated_coverage.csv',index=False)
    (out/'run_config.yaml').write_text(json.dumps(dict(run_utc=datetime.now(timezone.utc).isoformat(),raw=str(raw),new_drawn=28,unable=1,old_drawn=57,user_submission='annotator confirmed finished',geometry_processing='unchanged sr08h implementation, including Polygon.buffer(0) and existing smoothing',no_original_coordinates_modified=True),indent=2))
if __name__=='__main__':main()
