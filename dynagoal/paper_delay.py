"""Evaluate only Fast-JEPA for Fig.4 (DynaGoal-Delay); export results without plotting."""
import argparse,csv,json,os,subprocess,sys,statistics
from pathlib import Path
from .evaluate import load_settings
from .deployment import write_json
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(ROOT/'inference'),str(ROOT)])
    selection=load_settings()['paper_delay_curves']
    for task,cfg in selection.items():
        for seed in cfg['training_seeds']:
            checkpoint=a.checkpoint_root.resolve()/f'fast_{task}_s{seed}_e1/weights_epoch_1.pt'
            cmd=[sys.executable,'-m','dynagoal.evaluate','--subset','delay','--task',task,
                '--training-seed',str(seed),'--evaluation-seed',str(cfg['evaluation_seed']),
                '--checkpoint',str(checkpoint),'--output',str(a.output.resolve()/f'{task}_s{seed}')]
            print(cmd,flush=True)
            if not a.dry_run:subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    if a.dry_run:return
    rows=[]
    for task,cfg in selection.items():
        for delay in load_settings()['delay_ms']:
            cells=[json.loads((a.output/f'{task}_s{seed}'/f'delay_{delay}'/'SUMMARY.json').read_text()) for seed in cfg['training_seeds']]
            assert all(x['count']==100 and not x['smoke'] for x in cells)
            for seed,cell in zip(cfg['training_seeds'],cells):
                assert cell['task']==task and cell['training_seed']==seed
                assert cell['subset']=='delay' and cell['delay_ms']==delay
                assert cell['evaluation_seed']==cfg['evaluation_seed']
                expected=load_settings()['delay_cases'][f"{task}_e{cfg['evaluation_seed']}"]
                assert cell['case_manifest_sha256']==expected['case_sha256']
            values=[x['success_rate_percent'] for x in cells]
            rows.append(dict(task=task,delay_ms=delay,success_rate=statistics.mean(values),
                sample_sd=statistics.stdev(values) if len(values)>1 else None,
                training_seeds=cfg['training_seeds'],evaluation_seed=cfg['evaluation_seed']))
    write_json(a.output/'PAPER_CURVES.json',dict(selection_provenance='Historical per-task exploratory selection, not a paired common cohort',rows=rows))
    with (a.output/'PAPER_CURVES.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['| Task | Train seeds | Evaluation seed | 0 ms | 300 ms | 500 ms | 800 ms |',
        '|---|---|---:|---:|---:|---:|---:|']
    for task,cfg in selection.items():
        values=[]
        for delay in load_settings()['delay_ms']:
            row=next(x for x in rows if x['task']==task and x['delay_ms']==delay)
            value=f"{row['success_rate']:.2f}%"
            if row['sample_sd'] is not None:value+=f" ± {row['sample_sd']:.2f}"
            values.append(value)
        seeds='/'.join(map(str,cfg['training_seeds']))
        lines.append(f"| {task} | {seeds} | {cfg['evaluation_seed']} | "+' | '.join(values)+' |')
    text='\n'.join(lines)+'\n'
    (a.output/'PAPER_RESULTS.md').write_text(text)
    print(text)

if __name__=='__main__':main()
