import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import math
import torch.optim as optim
from auto_attack.autopgd_base import L1_projection
#from .l1_projection import project_onto_l1_ball
from .LPIPS_projection import project_onto_LPIPS_ball


l1_quantile = 0.99

def normalize_perturbation(perturbation, p):
    if p in ['inf', 'linf', 'Linf']:
        return perturbation.sign()
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        bs = perturbation.shape[0]
        pert_flat = perturbation.view(bs, -1)
        pert_normalized = F.normalize(pert_flat, p=2, dim=1)
        return pert_normalized.view_as(perturbation)
    elif p in [1, 1.0, 'l1', 'L1', '1']:
        bs = perturbation.shape[0]

        spatial_threhsoliding = True
        apply_sign = False

        if spatial_threhsoliding:
            pert_channels_summed = torch.sum(perturbation.abs(), dim=1)
            pert_channels_summed_flat = pert_channels_summed.view(bs,-1)
            dim = pert_channels_summed_flat.shape[1]
            threshold_idx = int(dim * l1_quantile)
            sort_elements= torch.sort(pert_channels_summed_flat, dim=1, descending=False)[0]
            threshold_element = sort_elements[torch.arange(bs), threshold_idx]
            remove_idcs = (pert_channels_summed < threshold_element[:,None,None]).unsqueeze(1)
            perturbation = perturbation * (~remove_idcs)
            pert_flat = perturbation.view(bs, -1)

            if apply_sign:
                raise NotImplementedError()
            else:
                pert_flat = F.normalize(pert_flat, p=1, dim=1)
            return pert_flat.view_as(perturbation)
        else:
            pert_flat = perturbation.view(bs, -1)
            dim = pert_flat.shape[1]
            threshold_idx = int(dim * l1_quantile)
            num_non_zero = dim - threshold_idx
            sort_elements= torch.sort(pert_flat.abs(), dim=1, descending=False)[0]
            threshold_element = sort_elements[torch.arange(bs), threshold_idx]
            remove_idcs = pert_flat.abs() < threshold_element.unsqueeze(1)

            if apply_sign:
                pert_flat = (1. / num_non_zero) * pert_flat.sign()
                pert_flat[remove_idcs] = 0.0
            else:
                pert_flat[remove_idcs] = 0.0
                pert_flat = F.normalize(pert_flat, p=1, dim=1)
            return pert_flat.view_as(perturbation)
    else:
        raise NotImplementedError('Normalization only supports l2 and inf norm')


def project_perturbation(perturbation, eps, p, center=None):
    if p in ['inf', 'linf', 'Linf']:
        pert_normalized = torch.clamp(perturbation, -eps, eps)
        return pert_normalized
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        #print('l2 renorm')
        pert_normalized = torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
        return pert_normalized
    elif p in [1, 1.0, 'l1', 'L1', '1']:
        ##pert_normalized = project_onto_l1_ball(perturbation, eps)
        ##return pert_normalized
        pert_normalized = L1_projection(center, perturbation, eps)
        return perturbation + pert_normalized
    elif p in ['LPIPS']:
        pert_normalized = project_onto_LPIPS_ball(perturbation, eps)
    else:
        raise NotImplementedError('Projection only supports l1, l2 and inf norm')


def reduce(loss, reduction) :
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError('reduction not supported')

#############################################iterative PGD attack
def logits_diff_loss(out, y_oh, reduction='mean'):
    #out: density_model output
    #y_oh: targets in one hot encoding
    #confidence:
    out_real = torch.sum((out * y_oh), 1)
    out_other = torch.max(out * (1. - y_oh) - y_oh * 1e13, 1)[0]

    diff = out_other - out_real

    return reduce(diff, reduction)

def conf_diff_loss(out, y_oh, reduction='mean'):
    #out: density_model output
    #y_oh: targets in one hot encoding
    #confidence:
    confidences = F.softmax(out, dim=1)
    conf_real = torch.sum((confidences * y_oh), 1)
    conf_other = torch.max(confidences * (1. - y_oh) - y_oh * 1e13, 1)[0]

    diff = conf_other - conf_real

    return reduce(diff, reduction)

def confidence_loss(out, y, reduction='mean'):
    confidences = F.softmax(out, dim=1)
    confidences_y = confidences[torch.arange(0, confidences.shape[0]), y]
    return reduce(confidences_y, reduction)

###################################
def create_early_stopping_mask(out, y, conf_threshold, targeted):
    finished = False
    conf, pred = torch.max(torch.nn.functional.softmax(out, dim=1), 1)
    conf_mask = conf > conf_threshold
    if targeted:
        correct_mask = torch.eq(y, pred)
    else:
        correct_mask = (~torch.eq(y, pred))

    mask = 1. - (conf_mask & correct_mask).float()

    if sum(1.0 - mask) == out.shape[0]:
        finished = True

    mask = mask[(..., ) + (None, ) * 3]
    return finished, mask

def initialize_perturbation(x, eps, norm, x_init=None, noise_generator=None):
    if x_init is None:
        pert = torch.zeros_like(x)
    else:
        pert = x_init - x

    if noise_generator is not None:
        pert += noise_generator(x)

    pert = project_perturbation(pert, eps, norm)
    return pert