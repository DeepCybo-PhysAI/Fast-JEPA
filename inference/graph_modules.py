
import time
import torch
from torch import nn

def copy_value(dst,src):
    target,source=dst,src
    for dim,(size,stride) in enumerate(zip(dst.shape,dst.stride())):
        if size>1 and stride==0:
            target=target.narrow(dim,0,1);source=source.narrow(dim,0,1)
    target.copy_(source)

def tensor_copy(x):
    out=torch.empty_strided(x.shape,x.stride(),device=x.device,dtype=x.dtype)
    copy_value(out,x)
    return out

def output_copy(x):
    if torch.is_tensor(x):return tensor_copy(x)
    if isinstance(x,dict):
        values={k:output_copy(v) for k,v in x.items()}
        return values if type(x) is dict else type(x)(**values)
    if isinstance(x,tuple):return tuple(output_copy(v) for v in x)
    if isinstance(x,list):return [output_copy(v) for v in x]
    return x

class CapturedModule(nn.Module):
    def __init__(self,inner):
        super().__init__();self.inner=inner;self.entries={};self.capture_seconds=0.;self.replays=0
        self.eval().requires_grad_(False)
    def __getattr__(self,name):
        try:return super().__getattr__(name)
        except AttributeError:
            if name=='inner':raise
            return getattr(super().__getattr__('inner'),name)
    @torch.inference_mode()
    def forward(self,*args,**kwargs):
        if self.training or self.inner.training:raise RuntimeError('Graphs require eval mode')
        if not args or not all(torch.is_tensor(x) and x.is_cuda for x in args):
            raise ValueError('Positional arguments must be CUDA tensors')
        if any(torch.is_tensor(v) for v in kwargs.values()):raise ValueError('Tensor kwargs unsupported')
        key=(tuple((tuple(x.shape),tuple(x.stride()),str(x.dtype),str(x.device)) for x in args),tuple(sorted(kwargs.items())))
        if key not in self.entries:
            start=time.perf_counter();static=tuple(tensor_copy(x) for x in args)
            stream=torch.cuda.Stream(device=args[0].device);stream.wait_stream(torch.cuda.current_stream(args[0].device))
            with torch.cuda.stream(stream):
                for _ in range(3):self.inner(*static,**kwargs)
            torch.cuda.current_stream(args[0].device).wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):out=self.inner(*static,**kwargs)
            torch.cuda.synchronize(args[0].device)
            self.entries[key]=(static,out,graph);self.capture_seconds+=time.perf_counter()-start
        static,out,graph=self.entries[key]
        for dst,src in zip(static,args):copy_value(dst,src)
        graph.replay();self.replays+=1
        return output_copy(out)

class CapturedActor(CapturedModule):
    def action_mean(self,z,intent,previous):return self(z,intent,previous)[0]

def install(model,mode='all'):
    names=['predictor'] if mode=='predictor' else ['encoder','projector','predictor','pred_proj','action_encoder','intent_actor']
    for name in names:
        if isinstance(getattr(model,name),CapturedModule):raise ValueError('Already wrapped')
        cls=CapturedActor if name=='intent_actor' else CapturedModule
        setattr(model,name,cls(getattr(model,name)))
    model.eval().requires_grad_(False)
    return model

def report(model):
    return {name:{'graphs':len(module.entries),'capture_seconds':module.capture_seconds,'replays':module.replays}
            for name,module in model.named_children() if isinstance(module,CapturedModule)}
