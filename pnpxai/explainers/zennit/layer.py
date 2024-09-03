'''Additional Utility Layers'''
import torch
from zennit.layer import Sum

class StackAndSum(torch.nn.Module):
    '''Compute the sum along an axis.

    Parameters
    ----------
    dim : int
        Dimension over which to sum.
    '''
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.zennit_sum = Sum(dim=dim)

    def forward(self, a, b):
        out = torch.stack([a, b.expand(a.size())], dim=self.dim)
        '''Computes the sum along a dimension.'''
        return self.zennit_sum(out)
