import torch
import math
import numpy as np
import torch.nn.functional as F

def maxlin(x_orig, w_orig, eps, p):
    ''' solves the optimization problem, for x in [0, 1]^d and p > 1,

    max <w, delta> s.th. ||delta||_p <= eps, x + delta \in [0, 1]^d
    '''
    bs = x_orig.shape[0]
    small_const = 1e-10
    x = x_orig.view(bs, -1)
    w = w_orig.view(bs, -1)
    gamma = x * (w < 0.) + (1. - x) * (w > 0.)
    delta = gamma.clone()

    w = w.abs()

    ind = gamma == 0.
    gamma_adj, w_adj = gamma.clone(), w.clone()
    gamma_adj[ind] = small_const
    w_adj[ind] = 0.

    mus = w_adj / (p * (gamma_adj ** (p - 1)))
    print('mus nan in tensor', mus.isnan().any())
    mussorted, ind = mus.sort(dim=1)
    gammasorted, wsorted = gamma.gather(1, ind), w_adj.gather(1, ind)

    # print(mussorted[-1])

    gammacum = torch.cat([torch.zeros([bs, 1], device=x.device),
                          (gammasorted ** p).cumsum(dim=1)],  # .fliplr()
                         # torch.zeros([bs, 1], device=x.device),
                         dim=1)
    gammacum = (gammasorted ** p).sum(dim=-1, keepdim=True) - gammacum
    wcum = (wsorted ** (p / (p - 1))).cumsum(dim=1)

    # print(gammacum[-1]) #wcum[-1]
    mussorted[mussorted==0] = small_const
    mucum = torch.cat([torch.zeros([bs, 1], device=x.device),
                       wcum / (p * mussorted) ** (p / (p - 1))], dim=1)
    print('mucum is nan', mucum.isnan().any())
    fs = gammacum + mucum - eps ** p
    # print(fs[-1], gammacum[-1], mucum[-1])

    ind = fs[:, 0] > 0.  # * (fs[-1] < 0.)
    # print(ind)
    lb = torch.zeros(bs).long()
    ub = lb + fs.shape[1]

    u = torch.arange(bs)
    for c in range(math.ceil(math.log2(fs.shape[1]))):
        a = (lb + ub) // 2
        indnew = fs[u, a] > 0.
        lb[indnew] = a[indnew].clone()
        ub[~indnew] = a[~indnew].clone()

    # print(lb, ub)
    pmstar = wcum[u, lb - 1] / (eps ** p - gammacum[u, lb])  # wcum[u, lb]
    print('pmstar is nan', pmstar.isnan().any())
    pmstar[pmstar == 0] = small_const
    deltamax = w ** (1 / (p - 1)) / pmstar.unsqueeze(1) ** (1 / p)  # ** (1 / (p - 1))
    print('deltamax is nan', deltamax.isnan().any())
    # print(deltamax)
    delta[ind] = torch.min(delta[ind],  # deltamax[ind].unsqueeze(1
                           # ) * torch.ones_like(delta[ind])
                           deltamax[ind])

    return delta.view(w_orig.shape) * w_orig.sign()


def LMO(grad, x_0, eps, p=2):
    n = x_0.shape[0]
    # -grad as we had -loss in the beginning
    if p == 2:
        return -eps*(grad)/grad.view(n, -1).norm(p=p, dim=1).view(-1, 1, 1, 1) + x_0
    elif p == 1:
        # ToDo bring m calculation outside to make more efficient
        m_ = grad.view(n, -1).shape[1]
        vals, idx = grad.view(n, -1).max(dim=1)
        return -eps*(F.one_hot(idx, num_classes=m_)*vals.sign().view(-1, 1)).view_as(grad) + x_0
    elif 1 < p < np.float('inf'):
        return (-eps*grad.sign()*grad.abs()**(1/(p-1)) / ((grad.abs()**(p/(p-1))).view(n, -1).sum(1)**(1/p)).view(-1, 1, 1, 1)) + x_0
