"""Frozen TCGA-only ×8 training with deterministic sampling and immutable recoverable snapshots."""
from pathlib import Path
import os,sys,time,json,hashlib,signal,argparse,traceback,platform,importlib.metadata
from datetime import datetime,timezone
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sr09_scale65_core import build,prepare,tensor

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume');args=ap.parse_args()
    root=Path(os.environ['PROJECT_ROOT']).resolve();c=json.loads((root/'results/sr09/scale65_rebuild_2026-10-27/train_config.json').read_text());out=root/c['output_dir']
    if args.resume:assert out.exists()
    else:out.mkdir(parents=True,exist_ok=False)
    os.nice(max(0,c['nice']-os.getpriority(os.PRIO_PROCESS,0)))
    torch.set_num_threads(c['threads']);torch.set_num_interop_threads(c['threads']);cv2.setNumThreads(c['threads']);torch.manual_seed(c['seed'])
    stopped=False;pause_total=0.;start=time.monotonic();global_best=float('inf');best_path=None;step=0;metrics=[]
    def stopping(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    def atomic_json(path,data):
        temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2));os.replace(temp,path)
    def status(phase,**kw):atomic_json(out/'progress.json',dict(phase=phase,time_utc=datetime.now(timezone.utc).isoformat(),step=step,iterations=c['iterations'],elapsed_seconds=time.monotonic()-start,load_wait_seconds=pause_total,load_1min=os.getloadavg()[0],**kw))
    def load_gate():
        nonlocal pause_total
        while os.getloadavg()[0]>c['load_threshold'] and not stopped:
            status('waiting_for_low_load');time.sleep(10);pause_total+=10
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    data=root/c['data_dir'];train=sorted((data/'hr/train').glob('*.png'));val=sorted((data/'hr/val').glob('*.png'))[:c['n_val']]
    assert len(train)>=4 and len(val)==200
    tilemeta=pd.read_csv(data/'tiles.csv').set_index('tile_id')
    assert set(tilemeta.loc[[p.stem for p in train],'patient']).isdisjoint(set(tilemeta.loc[[p.stem for p in val],'patient']))
    original_hash=digest(Path(c['weights_dir'])/c['weights_file'])
    if not args.resume:
        config=dict(config=c,start_utc=datetime.now(timezone.utc).isoformat(),initialization='official pretrained weights only; quick checkpoint not used',official_weights_sha256=original_hash,protocol_sha256=digest(root/c['protocol']),script_sha256=digest(__file__),core_sha256=digest(root/'scripts/sr09_scale65_core.py'),split_sha256=digest(data/'split.csv'),tiles_sha256=digest(data/'tiles.csv'),python=platform.python_version(),versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas']},train_count=len(train),val_count=len(val),resume_command='python scripts/sr09_scale65_train.py --resume <this-run-checkpoint>')
        (out/'run_config.yaml').write_text(json.dumps(config,indent=2))
        (out/'README.md').write_text('# Frozen ×8 formal training\n\nTCGA-only, official ×8 initialization, 15000 steps, batch4, L1; fixed first200 validation tile minimum L1 selects model. No quick weights or independent tests. All historical checkpoints/validation rows are preserved; progress/selection are live pointers. Full checkpoints support resume. End-of-run best.pth is a hard link to selected immutable best_iter state. See run_config.yaml, validation_manifest.csv, training_manifest.jsonl and metrics.csv.\n')
    else:
        prior=json.loads((out/'run_config.yaml').read_text());assert prior['config']==c and prior['official_weights_sha256']==original_hash
    status('preparing_fixed_validation');lrs=[];hrs=[];vrows=[];vd=out/'validation_inputs';vd.mkdir(exist_ok=True)
    for i,path in enumerate(val):
        lr,hr,params=prepare(path,c['seed'],1,0,i,False);dst=vd/f'{path.stem}.png'
        if dst.exists():assert np.array_equal(cv2.cvtColor(cv2.imread(str(dst)),cv2.COLOR_BGR2RGB),lr)
        else:assert cv2.imwrite(str(dst),cv2.cvtColor(lr,cv2.COLOR_RGB2BGR))
        vrows.append(dict(tile_id=path.stem,hr_path=str(path),hr_sha256=digest(path),lr_path=str(dst),lr_sha256=digest(dst),params=json.dumps(params)))
        lrs.append(lr);hrs.append(hr)
    if not args.resume:pd.DataFrame(vrows).to_csv(out/'validation_manifest.csv',index=False)
    vx,vy=tensor(lrs),tensor(hrs);load_gate()
    if stopped:status('stopped_before_model');return
    print('Loading official ×8 model',flush=True);model=build(c,pretrained=not bool(args.resume)).to('mps')
    opt=torch.optim.Adam(model.parameters(),lr=c['lr'],betas=tuple(c['betas']));sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c['iterations'],eta_min=c['eta_min'])
    if args.resume:
        rp=Path(args.resume).resolve();assert rp.is_relative_to((out/'checkpoints').resolve())
        saved=torch.load(rp,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler']);step=saved['step'];global_best=saved['best_l1'];best_path=saved['best_path'];metrics=saved['metrics']
        # Avoid rewinding evidence if later committed data exist.
        lines=(out/'training_manifest.jsonl').read_text().splitlines();logged=json.loads(lines[-1])['step'] if lines else 0
        assert logged==step,'Resume snapshot must match last committed step; use a new run/version for any rewind.'
    def validate():
        model.eval();loss_sum=0.;count=0
        for i in range(0,len(vx),c['batch']):
            load_gate()
            if stopped:raise InterruptedError('Stop during validation')
            with torch.no_grad():pred=model(vx[i:i+c['batch']].to('mps')).clamp(0,1);loss=F.l1_loss(pred,vy[i:i+c['batch']].to('mps')).item()
            loss_sum+=loss*len(pred);count+=len(pred);time.sleep(c['pause_seconds'])
        model.train();return loss_sum/count
    def checkpoint(tag):
        folder=out/'checkpoints';folder.mkdir(exist_ok=True);path=folder/f'{tag}_iter{step:06d}.pth';assert not path.exists()
        tmp=path.with_suffix('.tmp');torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),scheduler=sched.state_dict(),step=step,best_l1=global_best,best_path=best_path,metrics=metrics,seed=c['seed'],sampling='deterministic epoch permutation and step/sample seeds'),tmp);os.replace(tmp,path)
        atomic_json(out/'latest_checkpoint.json',dict(path=str(path),step=step));return path
    def evaluate_and_select(train_loss=None):
        nonlocal global_best,best_path
        status('validating');vl=validate();assert np.isfinite(vl)
        metrics.append(dict(step=step,train_l1=train_loss,val_l1=vl,lr=sched.get_last_lr()[0],elapsed_seconds=time.monotonic()-start))
        if vl<global_best:
            global_best=vl;bp=out/f'best_iter{step:06d}.pth';assert not bp.exists();torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},bp);best_path=str(bp)
        atomic_json(out/'selection.json',dict(best_l1=global_best,best_path=best_path,validated_step=step))
        # Each immutable validation record is evidence; CSV is a derived cumulative view.
        md=out/'validation_records';md.mkdir(exist_ok=True);(md/f'iter{step:06d}.json').write_text(json.dumps(metrics[-1],indent=2));pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        print(f'[{step}] train L1 {train_loss} | val L1 {vl:.6f} | best {global_best:.6f}',flush=True)
    try:
        if not args.resume:
            evaluate_and_select();checkpoint('checkpoint')
        elif step % c['val_every'] == 0 and (not metrics or metrics[-1]['step'] != step):
            evaluate_and_select()
        interval=[];batches=len(train)//c['batch'];cached_epoch=None;permutation=None
        status('training');training_start=time.monotonic();starting_step=step
        with (out/'training_manifest.jsonl').open('a') as manifest:
            while step<c['iterations'] and not stopped:
                load_gate()
                if stopped:break
                epoch=step//batches;offset=step%batches
                if epoch!=cached_epoch:permutation=np.random.default_rng([c['seed'],epoch]).permutation(len(train));cached_epoch=epoch
                indices=permutation[offset*c['batch']:(offset+1)*c['batch']];lrlist=[];hrlist=[];samples=[]
                for index in indices:
                    path=train[int(index)];lr,hr,params=prepare(path,c['seed'],0,step,int(index),True);lrlist.append(lr);hrlist.append(hr);samples.append(dict(tile_id=path.stem,index=int(index),params=params))
                x,y=tensor(lrlist).to('mps'),tensor(hrlist).to('mps');model.train();opt.zero_grad(set_to_none=True);pred=model(x);assert pred.shape==y.shape
                loss=F.l1_loss(pred,y);assert torch.isfinite(loss).item();loss.backward();opt.step();sched.step();step+=1;value=loss.item();interval.append(value)
                manifest.write(json.dumps(dict(step=step,epoch=epoch,samples=samples,train_l1=value))+'\n');manifest.flush()
                time.sleep(c['pause_seconds'])
                if step%c['progress_every']==0:
                    torch.mps.synchronize();eta=(time.monotonic()-training_start)/max(step-starting_step,1)*(c['iterations']-step)
                    status('training',recent_train_l1=float(np.mean(interval)),estimated_remaining_seconds=eta,mps_driver_bytes=torch.mps.driver_allocated_memory(),mps_allocated_bytes=torch.mps.current_allocated_memory())
                    print(f'[{step}/{c["iterations"]}] L1 {value:.6f} ETA training-only {eta/3600:.2f}h',flush=True)
                if step%c['val_every']==0 or step==c['iterations']:
                    evaluate_and_select(float(np.mean(interval)));interval=[];checkpoint('checkpoint')
            if stopped:
                checkpoint('interrupted');status('interrupted',best_l1=global_best);print('Stopped safely',flush=True)
            else:
                assert best_path and not (out/'best.pth').exists();os.link(best_path,out/'best.pth');status('complete',best_l1=global_best,best_path=best_path);print('Formal ×8 training complete',flush=True)
    except BaseException:
        (out/f'exception_iter{step:06d}.txt').write_text(traceback.format_exc())
        try:checkpoint('error')
        except Exception:pass
        status('failed');raise
if __name__=='__main__':main()
