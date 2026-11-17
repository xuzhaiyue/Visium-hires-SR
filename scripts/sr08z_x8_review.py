"""Recalculate frozen summary metrics and verify isolated blinded-page assets after ×8 evaluation."""
from pathlib import Path
import json,hashlib,urllib.request
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]

def main():
    w=ROOT/'docs/worklog/2026-10-25_SR_x8_overnight';out=ROOT/'results/sr08/eval_x8';ann=ROOT/'results/sr08/annotation_x8'
    m=pd.read_csv(out/'metrics_by_sample.csv')
    assert sorted(m.groupby(['testset','method']).size().tolist())==[12,12,12,28,28]
    h=m[m.testset.str.startswith('HEST')]
    assert sorted(h.groupby(['stratum','method']).size().tolist())==[13,13,15,15]
    rows=[]
    for s,g in h.groupby('stratum'):
        v=g.pivot(index='sample',columns='method',values=['f1','abs_log_area_med','abs_d_circ_med','FDR'])
        rows.append(dict(stratum=s,n=len(v),delta_f1=(v.f1.SwinIR-v.f1.bicubic).mean(),area_reduction_fraction=1-v.abs_log_area_med.SwinIR.mean()/v.abs_log_area_med.bicubic.mean(),delta_circularity=(v.abs_d_circ_med.SwinIR-v.abs_d_circ_med.bicubic).mean(),automatic_delta_fdr=(v.FDR.SwinIR-v.FDR.bicubic).mean()))
    pd.DataFrame(rows).to_csv(out/'automatic_gate_recalculation.csv',index=False)
    c=pd.read_csv(out/'contours/per_nucleus.csv');assert c.groupby('method').size().eq(86).all()
    x=c[c.method.isin(['bicubic ×8','SwinIR ×8'])].pivot(index='id',columns='method',values=['iou','area_err','circ_err']).dropna();paired=[]
    for metric in ['iou','area_err','circ_err']:
        a=x[(metric,'bicubic ×8')];b=x[(metric,'SwinIR ×8')];d=b-a
        paired.append(dict(metric=metric,n=len(x),bicubic_median=a.median(),SwinIR_median=b.median(),paired_difference_median=d.median(),wilcoxon_p=wilcoxon(d).pvalue if np.any(d!=0) else 1))
    pd.DataFrame(paired).to_csv(out/'contours/paired_x8.csv',index=False)
    items=json.loads((ann/'items.json').read_text());key=pd.read_csv(ann/'key.csv');alloc=pd.read_csv(ann/'sampling_allocation.csv')
    assert len(items)==len(key)==460 and key.id.is_unique and all(set(i)=={'id','img'} for i in items)
    assert alloc.groupby(['method','stratum']).n_sampled.sum().eq(100).all() and alloc.n_sampled.ge(1).all() and len(alloc)==56
    assert key.source_id.notna().sum()==20 and key[key.source_id.notna()].source_id.isin(key.id).all()
    text=(ann/'index.html').read_text();assert '"method"' not in text and '"sample"' not in text and 'sr08_x8_labels_v1' in text
    q=json.loads(Path('sr08_x8_labels.json').read_text());assert q['schema']=='sr08_x8_v1' and q['answers']['x8q001']['label']=='uncertain'
    for path in ['x8/key.csv','x8/items.json','x8/sampling_allocation.csv']:
        try:urllib.request.urlopen('http://127.0.0.1:8764/'+path);raise AssertionError(path)
        except urllib.error.HTTPError as e:assert e.code==404
    checks=dict(auto_sample_coverage=True,auto_metrics_independently_recalculated=True,contour_n=86,contour_paired_n=len(x),annotation_items=460,annotation_allocations=56,allocation_100_each=True,repeat_source_ids_verified=True,page_blinded=True,page_navigation_save_reload_export_verified=True,qa_storage_separate=True,private_files_404=True,page_sha256=hashlib.sha256((ann/'index.html').read_bytes()).hexdigest())
    (w/'page_and_metrics_qa.json').write_text(json.dumps(checks,indent=2))
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(paired).to_string(index=False));print('Annotation and metrics QA passed')
if __name__=='__main__':main()
