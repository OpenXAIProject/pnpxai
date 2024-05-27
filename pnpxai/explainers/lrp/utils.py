from torch.fx import Tracer
from torch.nn.modules import Module
from torch import nn

# a function letting the model work on zennit, listing all args up
def list_args_for_stack(*args):
    assert all(arg.shape[-2:] == args[0].shape[-2:] for arg in args), "Cannot convert"
    if all(arg.shape == args[0].shape for arg in args):
        return [arg for arg in args]
    
    # followings are experimental: matching dimension of all args
    ls = []
    dim = args[0].dim()
    max_d = max(arg.shape[0] for arg in args)
    for arg in args:
        if not arg.shape[0] in [1, max_d]:
            arg = arg.unsqueeze(0)
        if arg.shape[0] == 1:
            arg = arg.repeat(max_d, *[1 for _ in range(dim-1)])
        ls.append(arg)
    return ls

class LRPTracer(Tracer):
    # stop recursion when meeting timm's Attention
    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        return (
            (
                m.__module__.startswith("torch.nn")
                or m.__module__.startswith("torch.ao.nn")
                or (m.__module__ == "timm.models.vision_transformer" and m.__class__.__name__ == "Attention")
            ) and not isinstance(m, nn.Sequential)
        )
