"""Reproduce full Fast-JEPA on all task/weight pairs, keeping all measurements."""
import argparse, json, os, shlex, subprocess, sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
TASKS=['pusht','cube','reacher','tworoom']
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-seeds',nargs='+',type=int,default=[0,42,3072]);p.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    p.add_argument('--evaluation-seed',type=int,default=42);p.add_argument('--checkpoint-root',type=Path,default=Path(os.environ.get('STABLEWM_HOME', str(R/'cache')))/'checkpoints')
    p.add_argument('--output',type=Path,default=R/'outputs/main');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    py=os.environ.get('FAST_JEPA_PYTHON',str(R/'.venv/bin/python'))
    if len(a.training_seeds)!=len(set(a.training_seeds)):p.error('Repeated training seeds')
    for seed in a.training_seeds:
        for task in a.tasks:
            checkpoint=(a.checkpoint_root/f'fast_{task}_s{seed}_e1/weights_epoch_1.pt').resolve()
            out=a.output.resolve()/f'{task}_s{seed}_e{a.evaluation_seed}'
            if not a.dry_run:
                if out.exists() and any(out.iterdir()):raise FileExistsError(f'Existing cell {out}; use a new --output path')
                if not checkpoint.is_file():raise FileNotFoundError(f'{checkpoint}; train the checkpoint with scripts/train.sh or set --checkpoint-root')
                out.mkdir(parents=True)
            bench=[py,str(R/'inference/benchmark_paired.py'),'--checkpoint',str(checkpoint),'--mode','fast_guarded','--output',str(out/'benchmark.json')]
            evaluate=['bash',str(R/'scripts/eval.sh'),task,str(checkpoint),str(a.evaluation_seed)]
            print(shlex.join(bench),flush=True);print(shlex.join(evaluate),flush=True)
            if a.dry_run:continue
            env=os.environ.copy();env.update(PYTHONPATH=str(R/'inference'),FAST_JEPA_PYTHON=py,FAST_JEPA_RUN_ROOT=str(out),PAPER_EVAL_VIDEO_DIR=str(out/'videos'))
            with (out/'benchmark.log').open('w') as f:subprocess.run(bench,env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
            with (out/'evaluation.log').open('w') as f:subprocess.run(evaluate,env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
    if not a.dry_run:
        cmd=[py,str(R/'tools/aggregate_results.py'),str(a.output),'--training-seeds',*map(str,a.training_seeds),'--evaluation-seed',str(a.evaluation_seed),'--tasks',*a.tasks]
        subprocess.run(cmd,check=True)
if __name__=='__main__':main()
