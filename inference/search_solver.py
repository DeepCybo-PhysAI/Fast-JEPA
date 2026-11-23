
import time
import torch
from search_core import SearchCore


class SearchSolver(SearchCore):
    def __init__(self, *args, cache_encoding=True, prune_actor_terminal=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_encoding = bool(cache_encoding)
        self.prune_actor_terminal = bool(prune_actor_terminal)
        self.model.prune_actor_terminal = self.prune_actor_terminal
        self._encoding_cache = {}

    def _score(self, info, candidates):
        if not self.cache_encoding:
            return super()._score(info,candidates)
        if self.model.world_model_type != 'fast':
            raise ValueError('Cache path currently supports Fast only')

        key=tuple((k,v.data_ptr(),tuple(v.shape),str(v.dtype)) for k,v in info.items()
                  if torch.is_tensor(v) and k in ('pixels','goal'))
        if key not in self._encoding_cache:
            collapsed={k:v.to(device=self.device,dtype=self.dtype if v.is_floating_point() else None)
                       for k,v in info.items() if torch.is_tensor(v)}
            collapsed.pop('action',None)
            initial=self.model.encode(dict(collapsed))['emb'][:,-1:]
            goal=self.model._encode_goal(collapsed)
            self._encoding_cache[key]=(initial,goal)
        initial,goal=self._encoding_cache[key]
        b,n,h,a=candidates.shape
        expanded=initial[:,None].expand(-1,n,-1,-1).reshape(b*n,1,-1)
        predicted=self.model.predict_prefix(expanded,candidates.reshape(b*n,h,a),last_only=True)
        costs=(predicted.reshape(b,n,-1)-goal[:,None].detach()).square().sum(-1).float()
        if not torch.isfinite(costs).all():raise FloatingPointError('nonfinite search score')
        return costs

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        self._encoding_cache = {}
        predictor = self.model.predictor
        cuda = next(predictor.parameters()).is_cuda
        durations, events, starts = [], [], []
        def begin(module, inputs):
            if cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record(); starts.append(event)
            else:
                starts.append(time.perf_counter())
        def end(module, inputs, output):
            start = starts.pop()
            if cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record(); events.append((start,event))
            else:
                durations.append((time.perf_counter()-start)*1000)
        before = predictor.register_forward_pre_hook(begin)
        after = predictor.register_forward_hook(end)
        if cuda: torch.cuda.synchronize(next(predictor.parameters()).device)
        t0 = time.perf_counter()
        try:
            result = super().solve(info_dict, init_action)
            if cuda: torch.cuda.synchronize(next(predictor.parameters()).device)
            stats = result['timing']
            stats['synchronized_solve_ms'] = (time.perf_counter()-t0)*1000
            stats['wm_module_ms'] = sum(a.elapsed_time(b) for a,b in events) + sum(durations)
            stats['wm_module_calls'] = float(len(events)+len(durations))
            self.timing_history[-1].update(stats)
            return result
        finally:
            before.remove(); after.remove()
            self._encoding_cache = {}
