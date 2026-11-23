"""Run all twelve frozen checkpoints on a DynaGoal subset."""
import argparse,os,subprocess,sys
from pathlib import Path
from .evaluate import TASKS,load_settings
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset',choices=['shift','deadline','delay'],required=True)
    p.add_argument('--checkpoint-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(ROOT/'inference'),str(ROOT)])
    windows=load_settings()['delay_evaluation_seeds'] if a.subset=='delay' else [None]
    for seed in [0,42,3072]:
        for task in TASKS:
            for ev in windows:
                checkpoint=a.checkpoint_root.resolve()/f'fast_{task}_s{seed}_e1/weights_epoch_1.pt'
                name=f'{task}_s{seed}'+('' if ev is None else f'_e{ev}')
                cmd=[sys.executable,'-m','dynagoal.evaluate','--subset',a.subset,'--task',task,
                    '--checkpoint',str(checkpoint),'--training-seed',str(seed),
                    '--output',str(a.output.resolve()/name)]
                if ev is not None:cmd+=['--evaluation-seed',str(ev)]
                print(cmd,flush=True)
                if not a.dry_run:subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    if not a.dry_run:
        subprocess.run([sys.executable,'-m','dynagoal.aggregate',str(a.output.resolve()),'--subset',a.subset],cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
