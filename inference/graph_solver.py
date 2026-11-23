
import time
import torch
from search_solver import SearchSolver
from graph_modules import install,report
from sdpa_policy import sdpa_kernel_context

class GraphSearchSolver(SearchSolver):
    def __init__(self,*args,cuda_graph_mode='eager',**kwargs):
        super().__init__(*args,**kwargs)
        if cuda_graph_mode not in ['eager','predictor','all']:raise ValueError(cuda_graph_mode)
        self.cuda_graph_mode=cuda_graph_mode;self.graph_setup_seconds=0.
        if cuda_graph_mode!='eager':install(self.model,cuda_graph_mode)
    @torch.inference_mode()
    def configure(self,**kwargs):
        super().configure(**kwargs)
        if self.cuda_graph_mode=='eager':return


        start=time.perf_counter();device=next(self.model.parameters()).device
        with sdpa_kernel_context('math'):
            for batch in sorted(set([1,self.n_envs])):
                info={'pixels':torch.zeros(batch,1,3,224,224,device=device),
                      'goal':torch.zeros(batch,1,3,224,224,device=device),
                      'action':torch.zeros(batch,1,self.action_dim,device=device)}
                self.model.get_action(info,horizon=self.horizon)
            one={k:v[:1] for k,v in info.items()}
            self._score(one,torch.zeros(1,self.num_samples,self.horizon,self.action_dim,device=device))
            self._score(one,torch.zeros(1,1,self.horizon,self.action_dim,device=device))
        self._encoding_cache={};torch.cuda.synchronize(device)
        self.graph_setup_seconds=time.perf_counter()-start
    @torch.inference_mode()
    def solve(self,info_dict,init_action=None):
        before=sum(x['capture_seconds'] for x in report(self.model).values())
        result=super().solve(info_dict,init_action)
        after=sum(x['capture_seconds'] for x in report(self.model).values())
        result['timing']['graph_capture_seconds']=after-before
        self.timing_history[-1].update(result['timing'])
        return result
