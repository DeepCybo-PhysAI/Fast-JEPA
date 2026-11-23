"""Aggregate task-first paired training seeds; never average an incomplete grid silently."""
import argparse, csv, json, math, statistics
from pathlib import Path
TASKS=['pusht','cube','reacher','tworoom']

def stats(values):
    if not values or not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite/missing measurement')
    return {'mean':statistics.mean(values),'sample_sd':statistics.stdev(values) if len(values)>1 else None,
            'sample_variance':statistics.variance(values) if len(values)>1 else None,'n_training_seeds':len(values)}

def collect(root,seeds,evaluation_seed,tasks=TASKS):
    cells=[]
    for seed in seeds:
        for task in tasks:
            key=f'{task}_s{seed}_e{evaluation_seed}'
            p=root/key/'evaluation.txt.json'; b=root/key/'benchmark.json'
            d=json.loads(p.read_text()); t=json.loads(b.read_text())
            assert d['num_eval']==100 and d['evaluation_seed']==evaluation_seed,p
            assert d['checkpoint_audit']['training_seed']==seed,p
            success=d['metrics']['episode_successes'];assert len(success)==100
            assert (d['num_samples'],d['n_steps'],d['topk'])==(64,2,8)
            assert (d['horizon'],d['receding_horizon'],d['action_block'])==(5,5,5)
            assert t['mode']=='fast_guarded' and t['batch']==1 and t['warmups']==5 and t['repeats']==20
            sr=100*sum(bool(x) for x in success)/100
            assert abs(sr-float(d['metrics']['success_rate']))<1e-8
            cells.append({'task':task,'training_seed':seed,'evaluation_seed':evaluation_seed,'episodes':100,
                          'success_rate_percent':sr,'planning_ms':float(t['mean_ms']),
                          'graph_setup_seconds':float(t['graph_setup_seconds']),
                          'environment_evaluation_seconds':float(d['eval_total_time'])})
    taskwise={task:{metric:stats([c[metric] for c in cells if c['task']==task]) for metric in ['success_rate_percent','planning_ms']} for task in tasks}

    per_seed=[{metric:statistics.mean(c[metric] for c in cells if c['training_seed']==seed) for metric in ['success_rate_percent','planning_ms']} for seed in seeds]
    macro={metric:stats([row[metric] for row in per_seed]) for metric in ['success_rate_percent','planning_ms']}
    return {'training_seeds':seeds,'evaluation_seed':evaluation_seed,'tasks':tasks,'cells':cells,'taskwise':taskwise,'macro':macro,'partial_task_grid':tasks!=TASKS}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--training-seeds',nargs='+',type=int,default=[0,42,3072]);p.add_argument('--evaluation-seed',type=int,default=42);p.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS);a=p.parse_args()
    if len(set(a.training_seeds))!=len(a.training_seeds):p.error('Repeated training seeds')
    report=collect(a.root,a.training_seeds,a.evaluation_seed,a.tasks)
    (a.root/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.root/'RESULTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(report['cells'][0]));w.writeheader();w.writerows(report['cells'])
    rows=['| Task | SR (%), mean ± sample SD | Warm B1 planning (ms), mean ± sample SD |','|---|---:|---:|']
    def fmt(v):return f"{v['mean']:.2f} ± {v['sample_sd']:.2f}" if v['sample_sd'] is not None else f"{v['mean']:.2f} (one training seed)"
    for task in [*a.tasks,'Macro']:
        d=report['macro'] if task=='Macro' else report['taskwise'][task]
        rows.append(f"| {task} | {fmt(d['success_rate_percent'])} | {fmt(d['planning_ms'])} |")
    text='\n'.join(rows)+'\n';(a.root/'SUMMARY.md').write_text(text);print(text)
if __name__=='__main__':main()
