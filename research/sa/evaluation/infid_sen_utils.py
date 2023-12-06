#!pip install ttach

import torch
from torch.autograd import Variable
import copy
import numpy as np
# from explanations import GuidedBackpropReLUModel
import math

import lime
import captum
import os

FORWARD_BZ = 5000

# device = torch.device('cuda:5') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def forward_batch(model, input, batchsize):
    inputsize = input.shape[0]
    for count in range((inputsize - 1) // batchsize + 1):
        end = min(inputsize, (count + 1) * batchsize)
        if count == 0:
            tempinput = input[count * batchsize:end]
            out = model(tempinput.cuda().to(device))
            out = out.data.cpu().numpy()
        else:
            tempinput = input[count * batchsize:end]
            temp = model(tempinput.cuda().to(device)).data
            out = np.concatenate([out, temp.cpu().numpy()], axis=0)
    return out



def sample_eps_Inf(image, epsilon, N):
    images = np.tile(image, (N, 1, 1, 1))
    dim = images.shape
    return np.random.uniform(-1 * epsilon, epsilon, size=dim)


def get_explanation_pdt(image, model, label, exp, sg_r=None, sg_N=None, given_expl=None, binary_I=False):
    image_v = Variable(image, requires_grad=True)
    model =  model.to(device)
    model.zero_grad()
    out = model(image_v)
    pdtr = out[:, label]
    pdt = torch.sum(out[:, label])
    n_sample = 200
    
    if exp == 'Grad':
        pdt.backward()
        grad = image_v.grad
        expl = grad.data.cpu().numpy()
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'Int_Grad':
        for i in range(10):
            image_v = Variable(image * i/10, requires_grad=True)
            model.zero_grad()
            out = model(image_v)
            pdt = torch.sum(out[:, label])
            pdt.backward()
            grad = image_v.grad
            if i == 0:
                expl = grad.data.cpu().numpy() / 10
            else:
                expl += grad.data.cpu().numpy() / 10
        if binary_I:
            expl = expl * image.cpu().numpy()
    #added
    elif exp == 'LIME':
        expl = lime(image, label, model, n_sample)
    elif exp == 'LRP':
        expl = lrp(image, label, model, n_sample)
    elif exp == 'LRP2':
        expl = lrp2(image, label, model, n_sample)
    elif exp == 'IG':
        expl = int_grad(image, label, model, n_sample)
    elif exp == 'Grad_CAM':
        expl = grad_cam(image, label, model, n_sample)
    elif exp == 'Kernel_SHAP':
        expl = kernel_shap(image, label, model, n_sample)   
    elif exp == 'RAP':
        expl = rap(image, label, model, n_sample) 
    else:
        raise NotImplementedError('Explanation method not supported.')

    return expl, pdtr

#added lime, lrp, int_grad, grad_cam, kernel_shap
def lime(X, label, model, n_sample):
    from captum.attr import Lime
    from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
    from captum.attr._core.lime import get_exp_kernel_similarity_function
    from skimage.segmentation import quickshift, slic, watershed, felzenszwalb
    
    # generate superpixel
    X_ = X.squeeze(0).cpu().numpy().transpose(1,2,0)
    
    if X_.dtype != np.double:
        X_ = X_.astype(np.double)
        
    seg = felzenszwalb(X_, scale=250)
#     seg = quickshift(X_, ratio=1, kernel_size=5, max_dist=10) # default
#     seg = slic(X_, n_segments=150, compactness=10)

    seg = torch.LongTensor(seg).cuda().to(device)

#     exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)
#     explainer = Lime(model, SkLearnLinearRegression(), exp_eucl_distance)

    # define an explainer
    explainer = Lime(model)
    expl = explainer.attribute(
                            X, #X.unsqueeze(0),
                            target=label,
                            n_samples=n_sample,
                            feature_mask=seg,
#                             perturbations_per_eval=16,
                            ).squeeze(0)
    
    return expl.cpu().detach().numpy()

# def lrp(X, label, pdt, model, n_sample):
#     from captum.attr import LRP
#     from captum.attr._utils.lrp_rules import EpsilonRule
# #     model.layer.rule =  EpsilonRule()
#     explainer = LRP(model)
#     expl = explainer.attribute(X, target=label)
#     return expl.cpu().detach().numpy()

def lrp(X, label, model, n_sample):
    # https://github.com/kaifishr/PyTorchRelevancePropagation
    from src.lrp import LRPModel

    lrp_model = LRPModel(model)
    r = lrp_model.forward(X)

    return r.cpu().detach().numpy()

def lrp2(X, label,  model, n_sample):
    # https://github.com/wjNam/Relative_Attributing_Propagation/tree/master
    output = model(X)
    Res = model.relprop(R = output * label, alpha= 1).sum(dim=1, keepdim=True)
    return Res.cpu().detach().numpy()

def int_grad(X, label,  model, n_sample):
    from captum.attr import IntegratedGradients
    explainer = IntegratedGradients(model)
#     expl = explainer.attribute(X, target=label, n_steps = n_sample)
    expl = explainer.attribute(X, target=label)
    return expl.cpu().detach().numpy()

# def grad_cam(X, label, pdt, model, n_sample):
#     from captum.attr import GuidedGradCam
#     layers = []
#     for layer in model.children():
#         if len(list(layer.children())) == 0:
#             layers.append(layer)
# #     explainer = GuidedGradCam(model, layers[0])

#     explainer = GuidedGradCam(model, model.features[-1])
#     expl = explainer.attribute(X, target=label)
#     return expl.cpu().detach().numpy()

def grad_cam(X, label, model, n_sample):
    # https://github.com/jacobgil/pytorch-grad-cam
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    
    target_layers = [model.features[-1]]
    targets = [ClassifierOutputTarget(label)]
    explainer = GradCAM(model=model, target_layers=target_layers)
    expl = explainer(input_tensor=X, targets=targets)
    return expl
    
    
def kernel_shap(X, label, model, n_sample):
    from captum.attr import KernelShap
    from skimage.segmentation import quickshift, slic, watershed, felzenszwalb
    
    # generate superpixel
    X_ = X.squeeze(0).cpu().numpy().transpose(1,2,0)
    
    if X_.dtype != np.double:
        X_ = X_.astype(np.double)
        
    seg = felzenszwalb(X_, scale=250)
#     seg = quickshift(X_, ratio=1, kernel_size=5, max_dist=10) # default
#     seg = slic(X_, n_segments=150, compactness=10)

    seg = torch.LongTensor(seg).cuda().to(device)
    explainer = KernelShap(model)
    expl = explainer.attribute(
                            X, 
                            target=label,
                            n_samples = n_sample, 
                            feature_mask=seg,
                            perturbations_per_eval=16
                            )
    
    return expl.cpu().detach().numpy()

def rap(X, label, model, n_sample):
    # https://github.com/wjNam/Relative_Attributing_Propagation/tree/master
    expl = model.RAP_relprop(R=label)
    expl = (expl).sum(dim=1, keepdim=True)
#     heatmap = Res.permute(0, 2, 3, 1).data.cpu().numpy()
    return expl.cpu().detach().numpy()

def gen_pert(batch, size, pert):
    std, mean = torch.std_mean(batch)
#     print(std)
    if pert == "Gaussian":
        randd = torch.mul(torch.randn(size).cuda().to(device), std).add(mean)
        randd = torch.mul(randd,0.2) + batch
        randd = torch.minimum(batch, randd)
        randd = torch.maximum(batch - 1, randd)
        batch -= randd
        
        return batch, randd
        


def get_exp_infid(image, model, exp, label, pdt, binary_I, pert):
    num = 200
    image_tile = image.repeat(num,1,1,1)
    exp = torch.FloatTensor(exp).cuda().to(device)
    if pert == 'Gaussian':
        size = image_tile.size()
        image_copy, rand = gen_pert(image_tile, size, pert)        

    if pert == 'Gaussian' and not binary_I:
        exp_sum = torch.mul(rand, exp)
        exp_sum = exp_sum.sum(1).sum(1).sum(1)
        ks = np.ones(num)
    else:
        raise ValueError("Perturbation type and binary_I do not match.")
        

    image_v = Variable(image_tile, requires_grad=False)
    out = forward_batch(model, image_v, FORWARD_BZ)
    pdt_rm = (out[:, label])
    pdt_diff = pdt - pdt_rm

    # performs optimal scaling for each explanation before calculating the infidelity score
    exp_sum = exp_sum.cpu().numpy()
    beta = np.mean(ks*pdt_diff*exp_sum) / np.mean(ks*exp_sum*exp_sum)
    exp_sum *= beta
    infid = np.mean(ks*np.square(pdt_diff-exp_sum)) / np.mean(ks)
    
    del exp,image_v,image_tile

    return infid


def get_exp_sens(X, model, expl,exp, yy, pdt, sg_r,sg_N,sen_r,sen_N,norm,binary_I,given_expl):
    max_diff = -math.inf
    for _ in range(sen_N):
        sample = torch.FloatTensor(sample_eps_Inf(X.cpu().numpy(), sen_r, 1)).cuda().to(device)
        X_noisy = X + sample
        expl_eps, _ = get_explanation_pdt(X_noisy, model, yy, exp, sg_r, sg_N,
                                     given_expl=given_expl, binary_I=binary_I)
        max_diff = max(max_diff, np.linalg.norm(expl-expl_eps)/norm)
#     sample = None
    return max_diff

