"""Evaluate frozen Fast-JEPA weights on the original DynaGoal case manifests."""
import argparse, hashlib, json, os, platform, statistics, sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from . import deployment as dep
from .refined import RefinedRunner
from .profiles import ExploreRunner
from .cube import CubeRunner

ROOT=Path(__file__).resolve().parent
TASKS=('pusht','reacher','cube','tworoom')

def load_settings():
    return json.loads((ROOT/'settings.json').read_text())

def load_cases(condition):
    cfg=load_settings()['configs'][condition]
    path=ROOT/cfg['case_file']
    if dep.sha(path)!=cfg['case_sha256']:
        raise ValueError(f'Case checksum mismatch: {condition}')
    cases=json.loads(path.read_text())['cases']
    if len(cases)!=100 or len({c['case_id'] for c in cases})!=100:
        raise ValueError('Expected the fixed 100 distinct case identities')
    return cfg,cases

def load_delay_cases(task, evaluation_seed):
    settings=load_settings()
    entry=settings['delay_cases'][f'{task}_e{evaluation_seed}']
    path=ROOT/entry['case_file']
    if dep.sha(path)!=entry['case_sha256']:
        raise ValueError('Delay case checksum mismatch')
    cases=json.loads(path.read_text())['cases']
    if len(cases)!=100:
        raise ValueError('Expected the original 100 Delay cases')
    cfg=dict(settings['configs']['dynamic_'+task],start=evaluation_seed,**entry)
    return cfg,cases

def primary(condition,row,budget=None):
    if condition.startswith('deadline_'):
        return dep.deadline_success(row['success'],row['first_hit_step'],row['control_dt'],
            row['cumulative_compute_seconds'],budget)
    if condition=='dynamic_reacher':
        tr=row['trajectory']
        return any(x['success'] for x in tr if x['step']<75) and any(x['success'] for x in tr if x['step']>75)
    if condition=='dynamic_cube':
        return all(x['ever_success'] for x in row['goal_stages'])
    return all(x['success'] for x in row['trajectory'][-25:])

def execute(runner,condition,case,settings):
    if condition.startswith('deadline_'):
        return runner.episode(case,dep.Protocol('deadline',deadline_steps=settings['steps']))
    if condition=='dynamic_tworoom':
        return runner.dynamic_episode(case,settings['door_amplitude_px'],latency=True,
            steps=settings['steps'],feedback=settings['feedback_steps'])
    return runner.run_profile(case,settings['profile'])

def source_hashes():
    paths=list(ROOT.glob('*.py'))+list((ROOT.parent/'inference').glob('*.py'))
    paths+=list((ROOT.parent/'inference/vendor').glob('*.py'))
    return {p.relative_to(ROOT.parent).as_posix():dep.sha(p) for p in sorted(paths)}

def run(args):
    condition=('deadline_' if args.subset=='deadline' else 'dynamic_')+args.task
    settings,cases=load_delay_cases(args.task,args.evaluation_seed) if args.subset=='delay' else load_cases(condition)
    selected=cases[:1] if args.smoke else cases
    dest=args.output.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f'Refuse to overwrite existing output: {dest}')
    checkpoint=args.checkpoint.resolve()
    metadata=json.loads((checkpoint.parent/'run_metadata.json').read_text())
    if metadata['seed']!=args.training_seed:
        raise ValueError('Training seed does not match checkpoint metadata')
    expected=next(x['sha256'] for x in json.loads((ROOT/'checkpoint_hashes.json').read_text())
        if x['task']==args.task and x['training_seed']==args.training_seed)
    checkpoint_hash=dep.sha(checkpoint)
    if args.require_reference_checkpoint and checkpoint_hash!=expected:
        raise ValueError('Checkpoint differs from the archived paper checkpoint')
    import pygame
    if pygame.version.ver!='2.6.1' or getattr(pygame,'IS_CE',False):
        raise RuntimeError('Install the locked pygame 2.6.1 renderer using scripts/install.sh')
    torch.manual_seed(settings['start']);np.random.seed(settings['start']);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    cfg=OmegaConf.load(ROOT.parent/'inference/config/eval'/f'{args.task}.yaml')
    cfg.cache_dir=os.environ.get('LOCAL_DATASET_DIR') or os.environ.get('STABLEWM_HOME') or str(ROOT.parent/'cache')
    cfg.checkpoint=str(checkpoint)
    dataset=dep.get_dataset(cfg,cfg.eval.dataset_name)
    cls=dep.Runner if args.subset=='deadline' else RefinedRunner if args.task=='tworoom' else CubeRunner if args.task=='cube' else ExploreRunner
    dest.mkdir(parents=True,exist_ok=True)
    dep.write_json(dest/'cases.json',selected)
    provenance=dict(method='Fast-JEPA',condition=condition,subset=args.subset,
        task=args.task,training_seed=args.training_seed,evaluation_seed=args.evaluation_seed if args.subset=='delay' else None,checkpoint_sha256=checkpoint_hash,
        archived_checkpoint_match=checkpoint_hash==expected,case_manifest_sha256=settings['case_sha256'],
        settings=settings,smoke=args.smoke,gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__,pygame_version=pygame.version.ver,
        python_version=platform.python_version(),source_sha256=source_hashes(),
        precision=dict(matmul_precision='highest',allow_tf32=False,sdpa='math'),
        planning=dict(num_samples=64,n_steps=2,topk=8,cuda_graph='all',cache_encoding=True,prune_actor_terminal=True),
        timing_scope=load_settings()['timing'])
    with torch.inference_mode(),dep.sdpa_kernel_context('math'):
        runner=cls(args.task,'fast_guarded',cfg,dataset,'fast_only')
        try:
            if abs(runner.dt-settings['native_step_seconds'])>1e-10:
                raise ValueError('Environment native step duration changed')
            provenance['warmup']=runner.warmup(selected[0])
            provenance['native_gate']=runner.gate_native(selected[0])
            provenance['graph']=dep.graph_report(runner.model)
            if not provenance['native_gate']['pass_exact_actions'] or not provenance['graph']:
                raise RuntimeError('Native-loop/CUDA Graph gate failed')
            dep.write_json(dest/'provenance.json',provenance)
            original=runner.make_plan
            summaries=[]
            delays=load_settings()['delay_ms'] if args.subset=='delay' else [None]
            for delay in delays:
                timings=[]
                def make_plan(info,history):
                    actions,actual=original(info,history)
                    timings.append(actual)
                    return actions, actual if delay is None else delay/1000
                runner.make_plan=make_plan
                records=[]
                for i,case in enumerate(selected):
                    timings.clear()
                    row=execute(runner,condition,case,settings)
                    if args.subset!='deadline':
                        if len(row['trajectory'])!=100 or [p['observation_step'] for p in row['plans']]!=[0,25,50,75]:
                            raise RuntimeError('Dynamic scheduler changed the execution protocol')
                    if len(timings)!=len(row['plans']):
                        raise RuntimeError('Missing planning measurement')
                    if delay is not None:
                        for plan,actual in zip(row['plans'],timings):
                            plan['imposed_delay_seconds']=plan.pop('measured_seconds')
                            plan['actual_complete_plan_seconds']=actual
                    row['primary_success']=bool(primary(condition,row,settings.get('budget_seconds')))
                    row['training_seed']=args.training_seed
                    row['condition']=condition
                    row['delay_ms']=delay
                    folder=dest if delay is None else dest/f'delay_{delay}'
                    dep.write_json(folder/'episodes'/f'case_{i:03d}.json',row)
                    records.append(row)
                    print(json.dumps(dict(condition=condition,seed=args.training_seed,delay_ms=delay,
                        completed=i+1,total=len(selected),success=row['primary_success'])),flush=True)
                times=[p.get('actual_complete_plan_seconds',p.get('measured_seconds')) for row in records for p in row['plans']]
                summary=dict(condition=condition,subset=args.subset,task=args.task,
                    training_seed=args.training_seed,evaluation_seed=args.evaluation_seed if args.subset=='delay' else None,count=len(records),smoke=args.smoke,
                    success_rate_percent=100*sum(row['primary_success'] for row in records)/len(records),
                    planning_ms=1000*statistics.mean(times),plan_calls=len(times),delay_ms=delay,
                    checkpoint_sha256=checkpoint_hash,case_manifest_sha256=settings['case_sha256'])
                dep.write_json(folder/'SUMMARY.json',summary);summaries.append(summary)
            dep.write_json(dest/'COMPLETE.json',dict(status='PASS',summaries=summaries))
        finally:
            runner.world.close()
    return summaries

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset',choices=['shift','deadline','delay'],required=True)
    p.add_argument('--task',choices=TASKS,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--training-seed',type=int,choices=[0,42,3072],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evaluation-seed',type=int,choices=[0,42,3072],default=0,help='Original Delay evaluation window; Shift/Deadline use frozen per-condition cases')
    p.add_argument('--smoke',action='store_true',help='One fixed case only; excluded from formal aggregation')
    p.add_argument('--require-reference-checkpoint',action='store_true')
    run(p.parse_args())
if __name__=='__main__':main()
