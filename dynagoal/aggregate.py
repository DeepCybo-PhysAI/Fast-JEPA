"""Strict task-first aggregation for independently measured DynaGoal results."""
import argparse,csv,json,statistics
from pathlib import Path
from .evaluate import TASKS,load_cases,load_delay_cases,load_settings
from .deployment import write_json

def stats(xs):
    return dict(mean=statistics.mean(xs),sample_sd=statistics.stdev(xs),sample_variance=statistics.variance(xs))

def collect(root,subset):
    groups=load_settings()['delay_ms'] if subset=='delay' else [None]
    windows=load_settings()['delay_evaluation_seeds'] if subset=='delay' else [None]
    report={}
    for delay in groups:
        cells=[]
        for seed in [0,42,3072]:
            for task in TASKS:
                for ev in windows:
                    name=f'{task}_s{seed}'+('' if ev is None else f'_e{ev}')
                    p=root/name
                    complete=json.loads((p/'COMPLETE.json').read_text());assert complete['status']=='PASS'
                    if delay is not None:p=p/f'delay_{delay}'
                    x=json.loads((p/'SUMMARY.json').read_text())
                    condition=('deadline_' if subset=='deadline' else 'dynamic_')+task
                    assert (x['task'],x['subset'],x['training_seed'],x['delay_ms'])==(task,subset,seed,delay)
                    assert x['evaluation_seed']==ev
                    assert x['count']==100 and not x['smoke']
                    cfg,cases=load_cases(condition) if ev is None else load_delay_cases(task,ev)
                    assert x['case_manifest_sha256']==cfg['case_sha256']
                    episodes=[json.loads((p/'episodes'/f'case_{i:03d}.json').read_text()) for i in range(100)]
                    assert [v['case'] for v in episodes]==cases
                    assert x['success_rate_percent']==sum(bool(v['primary_success']) for v in episodes)
                    cells.append(x)
        # Delay first averages the three evaluation windows within each trained model.
        taskwise={task:{metric:stats([statistics.mean(x[metric] for x in cells
            if x['task']==task and x['training_seed']==seed) for seed in [0,42,3072]])
            for metric in ['success_rate_percent','planning_ms']} for task in TASKS}
        macro={metric:stats([statistics.mean(x[metric] for x in cells if x['training_seed']==seed)
            for seed in [0,42,3072]]) for metric in ['success_rate_percent','planning_ms']}
        report['measured' if delay is None else str(delay)]=dict(cells=cells,taskwise=taskwise,macro=macro)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--subset',choices=['shift','deadline','delay'],required=True);a=p.parse_args()
    result=collect(a.root,a.subset);write_json(a.root/'SUMMARY.json',result)
    lines=[]
    for delay,g in result.items():
        lines.extend([f'Condition: {a.subset}, response delay: {delay}',
            '| Task | SR (%), mean ± sample SD | Online planning (ms), mean ± sample SD |',
            '|---|---:|---:|'])
        for task in [*TASKS,'Macro']:
            x=g['macro'] if task=='Macro' else g['taskwise'][task]
            s=x['success_rate_percent'];t=x['planning_ms']
            lines.append(f"| {task} | {s['mean']:.2f} ± {s['sample_sd']:.2f} | {t['mean']:.2f} ± {t['sample_sd']:.2f} |")
        with (a.root/f'RESULTS_{delay}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(g['cells'][0]));writer.writeheader();writer.writerows(g['cells'])
    text='\n'.join(lines)+'\n';(a.root/'SUMMARY.md').write_text(text);print(text)
if __name__=='__main__':main()
