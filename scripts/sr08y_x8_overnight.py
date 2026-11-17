"""Authorized overnight audit and serial independent ×8 evaluation; never changes training selection."""
from pathlib import Path
import sys,os,time,json,hashlib,subprocess,zlib
from datetime import datetime,timezone
ROOT=Path(os.environ['PROJECT_ROOT']).resolve()
WORK=ROOT/'docs/worklog/2026-10-25_SR_x8_overnight'
RUN=ROOT/'results/sr08/models/x8/2026-10-24_run01'
PYTHON=sys.executable
BASE='real2_4.0_x8run01';SR=f'swinir8_{BASE}'
ENV=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTORCH_ENABLE_MPS_FALLBACK='1',CELLVIT_TORCH_THREADS='2',BATCH='2',CPUS='2',RAY_WORKERS='1',RAY_CPUS='1',MODEL='SAM')
os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))

def status(phase,**kw):
    data=dict(time_utc=datetime.now(timezone.utc).isoformat(),phase=phase,**kw)
    tmp=WORK/'status.tmp';tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,WORK/'status.json')
    print(data,flush=True)

def gate():
    while os.getloadavg()[0]>8:time.sleep(10)

def command(name,args):
    status(name)
    with (WORK/f'{name}.log').open('a') as f:subprocess.run(args,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    status('waiting_for_training')
    while True:
        state=json.loads((RUN/'progress.json').read_text())
        if state['phase']=='complete':break
        if state['phase'] in ['failed','interrupted','stopped_before_model']:raise RuntimeError('Training did not finish: '+state['phase'])
        # Confirm the actual training script is alive, not merely its stale status file.
        alive=subprocess.run(['pgrep','-f','^'+str(Path.home()/'miniforge3/envs/hemorph/bin/python')+' -u scripts/sr08v_train_x8.py$'],capture_output=True,text=True).stdout.strip()
        if not alive:raise RuntimeError('Training process absent before completion; preserve all evidence and recover explicitly')
        time.sleep(20)
    while subprocess.run(['pgrep','-f','^'+str(Path.home()/'miniforge3/envs/hemorph/bin/python')+' -u scripts/sr08v_train_x8.py$'],capture_output=True,text=True).stdout.strip():time.sleep(5)
    import cv2,numpy as np,pandas as pd,torch
    from sr08t_x8_core import build,tensor
    import sr08d_eval as E
    from sr02_degrade_benchmark import degrade_real2,save_png
    cv2.setNumThreads(2);torch.set_num_threads(2)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    status('training_audit');c=json.loads((ROOT/'config/sr_x8_train.yaml').read_text());cfg=json.loads((RUN/'run_config.yaml').read_text())
    metrics=pd.read_csv(RUN/'metrics.csv');selection=json.loads((RUN/'selection.json').read_text())
    assert state['step']==15000 and metrics.step.tolist()==list(range(0,15001,1000))
    assert np.isfinite(metrics.val_l1).all()
    best=RUN/'best.pth';chosen=Path(selection['best_path'])
    assert chosen.resolve().is_relative_to(RUN.resolve()) and digest(best)==digest(chosen)
    assert np.isclose(selection['best_l1'],metrics.val_l1.min(),rtol=0,atol=1e-12)
    assert cfg['protocol_sha256']==digest(ROOT/c['protocol']) and cfg['official_weights_sha256']==digest(Path(c['weights_dir'])/c['weights_file'])
    validation=pd.read_csv(RUN/'validation_manifest.csv');assert len(validation)==200 and validation.tile_id.is_unique
    meta=pd.read_csv(ROOT/c['data_dir']/'tiles.csv').set_index('tile_id');trainids=sorted(p.stem for p in (ROOT/c['data_dir']/'hr/train').glob('*.png'))
    assert set(meta.loc[trainids,'patient']).isdisjoint(set(meta.loc[validation.tile_id,'patient']))
    n=0;last=0
    with (RUN/'training_manifest.jsonl').open() as f:
        for line in f:
            r=json.loads(line);assert r['step']==last+1 and np.isfinite(r['train_l1']) and len(r['samples'])==4;last=r['step'];n+=1
    assert n==15000
    audit=dict(passed=True,n_steps=n,n_validation=len(validation),best_l1=selection['best_l1'],best_path=str(chosen),best_sha256=digest(best),protocol_sha256=cfg['protocol_sha256'],selection='TCGA minimum val L1 only; locked before independent test inference',checks=['15k consecutive finite training records','16 scheduled validations','200 fixed validation IDs','patient isolation','official initialization/protocol hashes','best state matches TCGA minimum'])
    (WORK/'training_audit.json').write_text(json.dumps(audit,indent=2))
    status('independent_degradation_and_inference');sets=E.test_sets();patches=pd.concat(sets.values(),ignore_index=True)
    degradation=[]
    for i,r in enumerate(patches.itertuples()):
        gtpath=E.BENCH/'patches'/r.sample/'gt'/f'{r.patch_id}.png'
        gt=cv2.cvtColor(cv2.imread(str(gtpath)),cv2.COLOR_BGR2RGB);assert gt.shape==(256,256,3)
        lr,params=degrade_real2(gt,32,np.random.default_rng([20261025,zlib.crc32(r.sample.encode()),zlib.crc32(r.patch_id.encode())]))
        dst=E.BENCH/'patches'/r.sample/BASE/f'{r.patch_id}.png'
        if dst.exists():assert np.array_equal(cv2.cvtColor(cv2.imread(str(dst)),cv2.COLOR_BGR2RGB),lr)
        else:save_png(dst,lr)
        degradation.append(dict(sample=r.sample,patch_id=r.patch_id,gt_path=str(gtpath),gt_sha256=digest(gtpath),lr_path=str(dst),lr_sha256=digest(dst),params=json.dumps(params)))
    pd.DataFrame(degradation).to_csv(WORK/'independent_input_manifest.csv',index=False)
    model=build(c,False);model.load_state_dict(torch.load(best,map_location='cpu',weights_only=True),strict=True);model.eval().to('mps')
    todo=[r for r in patches.itertuples() if not (E.BENCH/'patches'/r.sample/SR/f'{r.patch_id}.png').exists()]
    with torch.no_grad():
        for i in range(0,len(todo),4):
            gate();batch=todo[i:i+4];x=[cv2.cvtColor(cv2.imread(str(E.BENCH/'patches'/r.sample/BASE/f'{r.patch_id}.png')),cv2.COLOR_BGR2RGB) for r in batch]
            y=(model(tensor(x).to('mps')).clamp(0,1).cpu().numpy().transpose(0,2,3,1)*255+.5).astype(np.uint8)
            for r,im in zip(batch,y):save_png(E.BENCH/'patches'/r.sample/SR/f'{r.patch_id}.png',im)
            if i%100==0:status('independent_inference',done=i+len(batch),total=len(todo))
            time.sleep(.5)
    del model;torch.mps.empty_cache()
    status('mosaics')
    for method,p in sets.items():
        file=WORK/f'patches_{method}.csv';p.to_csv(file,index=False)
        command(f'mosaic_{method}',[PYTHON,'scripts/sr03a_build_mosaics.py','--method',method,'--patches',str(file),'--conditions',BASE,SR])
    for method in sets:
        for cond in [BASE,SR]:
            gate();ENV['TAG']=f'{method}_0.25mpp'
            command(f'cellvit_{method}_{cond}',['bash','scripts/03_run_cellvit.sh',f'sr_bench/mosaic/{method}_{cond}'])
    status('automatic_evaluation');E.lr_tag=lambda _:BASE;E.run_eval(8)
    command('registration',[PYTHON,'scripts/sr08x_x8_registration.py'])
    command('contours',[PYTHON,'-c',"import sys;from pathlib import Path;sys.path.insert(0,'scripts');import sr08h_contour_eval as h;h.C=Path('results/sr08/contours_x4_supplement_2026-10-23/final_analysis_2026-10-23/derived_bundle');h.OUT=Path('results/sr08/eval_x8/contours');h.CONDS={'native (upper bound)':'native','bicubic ×8':'real2_4.0_x8run01','SwinIR ×8':'swinir8_real2_4.0_x8run01'};h.main()"])
    command('annotation',[PYTHON,'scripts/sr08w_x8_annotation.py'])
    status('morning_report')
    report='# ×8 overnight training audit and automatic results\n\nTraining audit passed: 15000 consecutive finite steps, TCGA minimum validation L1 selection confirmed; independent test results were never used to choose weights.\n\n'+(ROOT/'results/sr08/eval_x8/VERDICT.md').read_text()+'\n\nHuman-corrected FDR and independent review remain pending. Existing 86 contours reused; ×8 category/area coverage still requires audit, and lack of significant paired contour accuracy improvement must not be hidden. The generated annotation page must undergo browser QA before user delivery. No true GEO hires or STPath run.\n'
    (WORK/'MORNING_REPORT.md').write_text(report)
    status('automatic_results_ready_pending_page_qa_and_human_labels')
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        (WORK/'exception.txt').write_text(traceback.format_exc());status('FAILED',error=str(exc));raise
