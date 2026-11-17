"""Preserve both raters and update ×8 slice-level FDR after independent review of uncertainty."""
from pathlib import Path
import sys,json,shutil,hashlib
from datetime import datetime,timezone
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/sr08/annotation_x8'
SEED=20261023
N_BOOT=20000

def main():
    src=Path(sys.argv[1]).resolve()
    out=BASE/'analysis_review_2026-10-25'
    out.mkdir(exist_ok=False)
    raw=BASE/'raw'/src.name
    if not raw.exists():shutil.copyfile(src,raw)
    payload=json.loads(raw.read_text())
    assert payload['schema']=='sr08_x8_v1' and raw.read_bytes()==src.read_bytes()
    review=payload['answers']
    first=pd.read_csv(BASE/'analysis_rater1_2026-10-25/labels_unblinded.csv')
    eligible=first[(~first.is_repeat)&first.label.eq('uncertain')]
    assert set(review)==set(eligible.id) and len(review)==27
    rev=pd.DataFrame([{'id':k,'review_label':v['label'],'review_time':v.get('time')} for k,v in review.items()])
    assert rev.review_label.isin(['yes','no','uncertain']).all()
    merged=first.merge(rev,on='id',how='left',validate='one_to_one')
    merged['final_label']=merged.label
    select=merged.review_label.notna()
    merged.loc[select,'final_label']=merged.loc[select,'review_label']
    merged['resolution_rule']='retain_first_rater'
    merged.loc[select,'resolution_rule']='independent_review_of_first_rater_uncertain'
    merged.to_csv(out/'labels_adjudicated.csv',index=False)
    rev.merge(eligible,on='id',validate='one_to_one').to_csv(out/'review_pairs.csv',index=False)
    originals=merged[~merged.is_repeat]
    unmatched=originals[originals['cat'].eq('unmatched')]
    stats=unmatched.groupby(['sample','method','stratum']).final_label.agg(n='size',n_no=lambda s:int(s.eq('no').sum()),n_uncertain=lambda s:int(s.eq('uncertain').sum())).reset_index()
    per=stats.merge(pd.read_csv(BASE/'denominators.csv'),on=['sample','method'],validate='one_to_one')
    per['fdr_main']=per.automatic_FDR*per.n_no/per.n
    per['fdr_conservative']=per.automatic_FDR*(per.n_no+per.n_uncertain)/per.n
    per.to_csv(out/'corrected_fdr_by_sample.csv',index=False)
    originals.groupby(['stratum','method','cat','final_label']).size().rename('n').reset_index().to_csv(out/'label_counts.csv',index=False)
    rng=np.random.default_rng(SEED); rows=[]
    for stratum,g in per.groupby('stratum'):
        wide=g.pivot(index='sample',columns='method',values=['fdr_main','fdr_conservative'])
        assert not wide.isna().any().any()
        idx=rng.integers(len(wide),size=(N_BOOT,len(wide)))
        for scenario in ['main','conservative']:
            col='fdr_'+scenario;b=wide[(col,'bicubic')].to_numpy();s=wide[(col,'SwinIR')].to_numpy()
            d=s-b;boot=d[idx].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
            rows.append(dict(stratum=stratum,scenario=scenario,n_samples=len(wide),bicubic_mean=b.mean(),SwinIR_mean=s.mean(),delta_mean=d.mean(),ci95_low=lo,ci95_high=hi,ci_upper_le_002=bool(hi<=.02),delta_le_002=bool(d.mean()<=.02)))
            np.save(out/f'bootstrap_{stratum}_{scenario}.npy',boot)
    summary=pd.DataFrame(rows);summary.to_csv(out/'corrected_fdr_summary.csv',index=False)
    unresolved=originals[originals.final_label.eq('uncertain')]
    unresolved.to_csv(out/'unresolved_items.csv',index=False)
    config=dict(run_utc=datetime.now(timezone.utc).isoformat(),review_source=str(src),review_archived=str(raw),review_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),seed=SEED,n_boot=N_BOOT,unit='paired slices within resolution stratum',review_n=27,independent_second_rater='confirmed by user; identity and experience pending',resolution='first uncertain → review label, otherwise preserve first; no spatial reassignment',n_unresolved=len(unresolved))
    (out/'run_config.yaml').write_text(json.dumps(config,indent=2))
    lines=['# ×8 independent review analysis','',f'27/27 review items complete. User confirms an independent second rater; identity and relevant experience not yet supplied.',f'Review outcomes: {rev.review_label.value_counts().to_dict()}. Remaining original uncertain items: {len(unresolved)}; unmatched subset: {int(unresolved["cat"].eq("unmatched").sum())}.','', '```',summary.to_string(index=False),'```','', 'Both raters preserved; only originally uncertain nonrepeat items take the independent review label. Repeat items are excluded from FDR. Frozen matching and thresholds unchanged. Slice-only paired percentile bootstrap (20000, seed 20261023) does not fully propagate within-slice label sampling uncertainty. Report both main and conservative scenarios with separate CIs. Do not interpret uncertainty sensitivity ranges as confidence intervals.','', 'Caveats retained: reused 86 contours still require 14 additions to cover ×8 category/area strata. Background controls are automatic-reference-negative, not manual negatives. Rater credentials will be added during manuscript preparation as requested by the user. No formal labels were supplied by the assistant.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__':main()
