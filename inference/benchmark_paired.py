
import argparse,json,time,statistics
from pathlib import Path
from types import SimpleNamespace
import torch
from gymnasium.spaces import Box
from diagnostics import load
from graph_solver import GraphSearchSolver
from sdpa_policy import sdpa_kernel_context
from runtime import configure_deterministic_math

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--mode',choices=['fast_guarded'],default='fast_guarded');p.add_argument('--output',required=True);a=p.parse_args()
 configure_deterministic_math(3072);model=load(a.checkpoint).requires_grad_(False);model.interpolate_pos_encoding=True
 kw=dict(batch_size=1,device='cuda',seed=0,var_scale=.25)
 s=GraphSearchSolver(model,num_samples=64,n_steps=2,topk=8,cache_encoding=True,prune_actor_terminal=True,cuda_graph_mode='all',**kw)
 s.configure(action_space=Box(-1,1,shape=(1,model.get_action_dim()//5)),n_envs=1,config=SimpleNamespace(horizon=5,action_block=5))
 info={'pixels':torch.randn(1,1,3,224,224,device='cuda'),'goal':torch.randn(1,1,3,224,224,device='cuda'),'action':torch.randn(1,1,model.get_action_dim(),device='cuda')}
 raw=[]
 with torch.inference_mode(),sdpa_kernel_context('math'):
  for i in range(25):
   s.torch_gen.manual_seed(0);torch.cuda.synchronize();t=time.perf_counter();s.solve(info);torch.cuda.synchronize()
   if i>=5:raw.append(1000*(time.perf_counter()-t))
 out={'mode':a.mode,'checkpoint':a.checkpoint,'batch':1,'warmups':5,'repeats':20,'raw_ms':raw,'mean_ms':statistics.mean(raw),'median_ms':statistics.median(raw),'graph_setup_seconds':getattr(s,'graph_setup_seconds',0.),'scope':'complete synchronized solver including Actor,CEM,CPU output copy; excludes preprocessing,simulation,loading and graph initialization','device':torch.cuda.get_device_name()}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
