import io
# import math
# import os, inspect, sys
# import pickle
# from functools import partial

# from pathlib import Path
# import gc
import time
# from blended_diffusion.optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR
# from blended_diffusion.resizer import Resizer
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.tensorboard.summary import hparams
# from blended_diffusion.utils_blended.metrics_accumulator import MetricsAccumulator
# from blended_diffusion.utils_blended.model_normalization import ResizeWrapper, ResizeAndMeanWrapper
# from blended_diffusion.utils_blended.video import save_video
import torch.nn as nn
import random
# from blended_diffusion.optimization.augmentations import ImageAugmentations
from PIL import Image


import torch
# import torchvision
from torchvision import transforms
# import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
# from torch.nn.functional import mse_loss
# from blended_diffusion.optimization.losses import range_loss, d_clip_loss
# import lpips
import numpy as np

#from blended_diffusion.CLIP import clip
from blended_diffusion.guided_diffusion.guided_diffusion.script_util import (
    #NUM_CLASSES,
    # model_and_diffusion_defaults,
    # classifier_defaults,
    create_model_and_diffusion,
    # create_classifier,
    #add_dict_to_argparser,
    #args_to_dict
)

import matplotlib.pyplot as plt
from blended_diffusion.utils_blended.visualization import show_tensor_image, show_editied_masked_image

#ACSM_dir = '/mnt/SHARED/valentyn/ACSM'
##ACSM_dir = '/scratch/vboreiko87/projects/ACSM'

##sys.path.insert(0, ACSM_dir)

##print(sys.path)

# from utils_svces.get_config import get_config
# from utils_svces.Evaluator import Evaluator


import robustness.datasets
import robustness.model_utils

def disp(bchw):
    img = np.transpose(bchw[0].detach().cpu().numpy(), (1,2,0))
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.imshow(img)
    plt.show()


from collections import OrderedDict
from typing import Tuple

class ImageNormalizer(nn.Module):
    def __init__(
        self, 
        mean: Tuple[float, float, float], 
        std: Tuple[float, float, float]
    ) -> None:
        super(ImageNormalizer, self).__init__()
        self.register_buffer("mean", torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # print("Madry input shape", input.shape)
        return (input - self.mean) / self.std


def normalize_model(
        model: nn.Module, 
        mean: Tuple[float, float, float], 
        std: Tuple[float, float, float]
    ) -> nn.Module:
    layers = OrderedDict([("normalize", ImageNormalizer(mean, std)), ("model", model)])
    return nn.Sequential(layers)

class EmptyWriter:
    def __init__(self):
        pass

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        pass

    def flush(self):
        pass

    def close(self):
        pass


# class CorrectedSummaryWriter(SummaryWriter):
#     def add_hparams(self, hparam_dict, metric_dict):
#         torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
#         if type(hparam_dict) is not dict or type(metric_dict) is not dict:
#             raise TypeError('hparam_dict and metric_dict should be dictionary.')
#         exp, ssi, sei = hparams(hparam_dict, metric_dict)

#         self.file_writer.add_summary(exp)
#         self.file_writer.add_summary(ssi)
#         self.file_writer.add_summary(sei)
#         for k, v in metric_dict.items():
#             self.add_scalar(k, v)

def cone_projection(grad_temp_1, grad_temp_2, deg):
    angles_before = torch.acos(
        (grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_1.norm(p=2, dim=1) * grad_temp_2.norm(p=2, dim=1)))
    ##print('angle before', angles_before)

    grad_temp_2 /= grad_temp_2.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
    grad_temp_1 = grad_temp_1 - ((grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_2.norm(p=2, dim=1) ** 2)).view(
        grad_temp_1.shape[0], -1) * grad_temp_2
    grad_temp_1 /= grad_temp_1.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
    # cone_projection = grad_temp_1 + grad_temp_2 45 deg
    radians = torch.tensor([deg], device=grad_temp_1.device).deg2rad()
    ##print('angle after', radians, torch.acos((grad_temp_1*grad_temp_2).sum(1) / (grad_temp_1.norm(p=2,dim=1) * grad_temp_2.norm(p=2,dim=1))))

    cone_projection = grad_temp_1 * torch.tan(radians) + grad_temp_2

    # second classifier is a non-robust one -
    # unless we are less than 45 degrees away - don't cone project
    grad_temp = grad_temp_2.clone()
    loop_projecting = time.time()
    grad_temp[angles_before > radians] = cone_projection[angles_before > radians]

    return grad_temp

def _map_img(x):
    return 0.5 * (x + 1)


def _renormalize_gradient(grad, eps, small_const=1e-22):
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    #print('grad norm is', grad_norm)
    grad_norm = torch.where(grad_norm < small_const, grad_norm+small_const, grad_norm)
    grad /= grad_norm
    grad *= eps.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    return grad, grad_norm

def compute_lp_dist(diff, p):
    diff_abs_flat = diff.abs().view(diff.shape[0], -1)
    if p == 1.0:
        lp_dist = torch.sum(diff_abs_flat, dim=1)
    else:
        lp_dist = torch.sum(diff_abs_flat**p, dim=1)
    return lp_dist

def compute_lp_gradient(diff, p, small_const=1e-12):
    if p < 1:
        grad_temp = (p * (diff.abs() + small_const) ** (p - 1)) * diff.sign()
    else:
        grad_temp = (p * diff.abs() ** (p - 1)) * diff.sign()
    return grad_temp

def min_max_scale(tensor):
    tensor_ = tensor.clone()
    tensor_ -= tensor_.view(tensor.shape[0], -1).min(1)[0].view(-1, 1, 1, 1)
    tensor_ /= tensor_.view(tensor.shape[0], -1).max(1)[0].view(-1, 1, 1, 1)
    return tensor_

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  init_image_pil = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
  init_image_pil = init_image_pil.resize((224, 224), Image.LANCZOS)  # type: ignore
  image = TF.to_tensor(init_image_pil).unsqueeze(0)
  #image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  #image = tf.expand_dims(image, 0)
  return image

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    # u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()

    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)

    inu = 2 * (indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)

    s1 = -u.sum(dim=1)

    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)

    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)

    if c2.nelement != 0:

        lb = torch.zeros_like(c2).float()
        ub = torch.ones_like(lb) * (bs.shape[1] - 1)

        # print(c2.shape, lb.shape)

        nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
        counter2 = torch.zeros_like(lb).long()
        counter = 0

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2.)
            counter2 = counter4.type(torch.LongTensor)

            c8 = s[c2, counter2] + c[c2] < 0
            ind3 = c8.nonzero().squeeze(1)
            ind32 = (~c8).nonzero().squeeze(1)
            # print(ind3.shape)
            if ind3.nelement != 0:
                lb[ind3] = counter4[ind3]
            if ind32.nelement != 0:
                ub[ind32] = counter4[ind32]

            # print(lb, ub)
            counter += 1

        lb2 = lb.long()
        alpha = (-s[c2, lb2] - c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
        d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])

    return (sigma * d).view(x2.shape)

def total_variation_loss(img):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def project_perturbation(perturbation, eps, p, center=None):
    if p in ['inf', 'linf', 'Linf']:
        pert_normalized = torch.clamp(perturbation, -eps, eps)
        return pert_normalized
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        print('l2 renorm')
        pert_normalized = torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
        return pert_normalized
    elif p in [1, 1.0, 'l1', 'L1', '1']:
        ##pert_normalized = project_onto_l1_ball(perturbation, eps)
        ##return pert_normalized
        pert_normalized = L1_projection(center, perturbation, eps)
        return perturbation + pert_normalized
    #elif p in ['LPIPS']:
    #    pert_normalized = project_onto_LPIPS_ball(perturbation, eps)
    else:
        raise NotImplementedError('Projection only supports l1, l2 and inf norm')

class DiffusionAttack():
    def __init__(self, args) -> None:
        self.args = args
        self.probs = None
        self.y = None
        self.writer = None
        self.small_const = 1e-12
        # self.tensorboard_counter = 0
        # self.verbose = args.verbose

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = {'image_size': self.args.model_output_size, # 256
                            #  'num_channels': 256, # for imagenet checkpoints/256x256_diffusion_uncond.pt
                             'num_channels' : 128, # for guided-diffusion-cxr C:\Users\lab402\Projects\guided-diffusion-cxr\LOGDIR_for_rsna_lr1e-4_3e-5\model230000.pt
                             'num_res_blocks': 2,
                             'resblock_updown': True,
                             'num_heads': 4,
                             'num_heads_upsample': -1,
                             'num_head_channels': 64,
                            #  'attention_resolutions': '32,16,8', # for imagenet checkpoints/256x256_diffusion_uncond.pt
                             'attention_resolutions' : '16, 8', # for guided-diffusion-cxr C:\Users\lab402\Projects\guided-diffusion-cxr\LOGDIR_for_rsna_lr1e-4_3e-5\model230000.pt
                             'channel_mult': '',
                             'dropout': 0.0,
                             'class_cond': False,
                             'use_checkpoint': False,
                             'use_scale_shift_norm': True,
                             'use_fp16': True, # True,
                             'use_new_attention_order': False,
                             'learn_sigma': True,
                             'diffusion_steps': 1000,
                             'noise_schedule': 'linear',
                             'timestep_respacing': "200", #self.args.timestep_respacing,
                             'use_kl': False,
                             'predict_xstart': False,
                             'rescale_timesteps': True,
                             'rescale_learned_sigmas': False}


        self.device = self.args.device
        print("Using device:", self.device)

        
        #########################################################
        ############### DIFFUSION MODELS ##############
        #########################################################
        
        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        ###### DIFFUSION FOR IMAGENET UNCOND ######
        # self.model.load_state_dict(
        #     torch.load(
        #         "checkpoints/256x256_diffusion_uncond.pt",
        #         map_location="cpu",
        #     )
        # )
        # self.model.requires_grad_(False).eval().to(self.device)

        # if args.device_ids is not None and len(args.device_ids) > 1:
        #     self.model = nn.DataParallel(self.model, device_ids=args.device_ids)

        # for name, param in self.model.named_parameters():
        #     if "qkv" in name or "norm" in name or "proj" in name:
        #         param.requires_grad_()
                
        # if self.model_config["use_fp16"]:
        #     self.model.convert_to_fp16()
        # args.device = self.device
        
        
        ###### CUSTOM DIFFUSION MODELS ######
        self.model.load_state_dict(
            torch.load(
                # r'checkpoints/RSNA_diffusion.pt',
                # r'/home/wonjun29/code/pnp/guided-diffusion/training_checkpoints_++1171_--11710_+-0_-+0/best_so_far/model260000.pt',
                # r'/home/wonjun29/code/pnp/guided-diffusion/training_checkpoints_++1171_--117100_+-0_-+0/model120000.pt',
                '/home/wonjun29/code/pnp/guided-diffusion/training_checkpoints_oldcode_1171_1171/model050000.pt',
                # '/home/wonjun/repos/pnp/guided-diffusion/training_checkpoints_oldcode_1171_1171/model050000.pt',
                map_location='cpu')
        )
        self.model.requires_grad_(False).eval().to(self.device)

        if args.device_ids is not None and len(args.device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=args.device_ids)

        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
                
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()
        args.device = self.device

        #########################################################
        ##################### CLASSIFIERS ####################
        #########################################################
        
        #### BOREIKO ET AL'S ROBUST CLASSIFIER #############
        # self.classifier_config = get_config(args)
        # evaluator = Evaluator(args, self.classifier_config, {}, None)
        # self.classifier = evaluator.load_model(
        #     self.args.classifier_type, prewrapper=partial(ResizeAndMeanWrapper, size=self.args.classifier_size_1, interpolation=self.args.interpolation_int_1)
        # )
        # self.classifier = self.classifier.model.module.model # not using TemperatureWrapper, DataParallel, ResizeAndMeanWrapper (thus only using ImageNormalizer and ResNet)
        # self.classifier.to(self.device)
        # self.classifier.eval()
        
        ## RECREATE BOREIKO ET AL'S ROBUST CLASSIFIER BUT LESS NESTING ######
        # classifier = torchvision.models.resnet50(pretrained=False).to(self.device)
        # self.classifier = normalize_model(model=classifier,
        #                                   mean=(0.485, 0.456, 0.406),
        #                                   std=(0.229, 0.224, 0.225))
        # state_dict = torch.load(r'.\ImageNet1000Models\Madry_l2_improved\checkpoint\ep_3.pt',
        #                         map_location='cpu')
        # self.classifier.model.load_state_dict(state_dict, strict=True)
        # self.classifier.to(self.device)
        # self.classifier.eval()
        
        
        ### DINO VIT, NON-ROBUST #############
        # vit = ViT_Classifier(
        #     vit_arch='vit_small',
        #     vit_patch_size=16,
        #     n_last_blocks=4,
        #     avgpool_patchtokens=False,
        #     vit_checkpoint_path=r'./dino_checkpoints/dino_deitsmall16_pretrain.pth',
        #     num_labels=1000,
        #     linear_clf_checkpoint_path=r'./dino_checkpoints/dino_deitsmall16_linearweights.pth'
        # )
        # self.classifier = normalize_model(model = vit,
        #                                   mean=(0.485, 0.456, 0.406),
        #                                   std=(0.229, 0.224, 0.225))
        # self.classifier.to(self.device)
        # self.classifier.eval()
        
        
        ######## CUSTOM RESNET-50, ROBUST ##########
        # data_path = r'E:\ILSVRC2012_img_train'
        # ds = robustness.datasets.CustomImageNet(
        #     data_path = data_path,
        #     # custom_grouping=[[wnid] for wnid in os.listdir(data_path+r'\train')]
        #     custom_grouping=[['n07745940'], ['n07734744']]
        # )
        # resume_path = r'C:\Users\lab402\Projects\DVCEs-windows\DVCEs_simpler_code\robust_training_l2_eps3_20advsteps_strawberry_mushroom\checkpoint.pt.best'
        # self.classifier, _ = robustness.model_utils.make_and_restore_model(arch='resnet50',
        #                                                                    dataset=ds,
        #                                                                    resume_path=resume_path)
        # self.classifier.to(self.device)
        # self.classifier.eval()
        
        ####### ROBUSTNESS PRETRAINED MODEL ######
        # data_path = '/content/drive/MyDrive/data/ImageNet'
        # ds = robustness.datasets.ImageNet(data_path=data_path)
        # # resume_path = r'C:\Users\lab402\Projects\DVCEs-windows\DVCEs_simpler_code\robustness_pretrained_models\imagenet_l2_3_0.pt'
        # resume_path = '/content/drive/MyDrive/Counterfactuals/DVCEs/robust_models/imagenet_l2_3_0.pt'
        # self.classifier, _ = robustness.model_utils.make_and_restore_model(arch='resnet50',
        #                                                                      dataset=ds,
        #                                                                     resume_path=resume_path)
        # self.classifier.to(self.device)
        # self.classifier.eval()
        
        ###### RSNA ROBUST RESNET50 ########
        # data_path = '/media/wonjun/HDD2TB/rsna-pneumonia-detection-jpgs/NORMAL+OPACITY'
        # ds = robustness.datasets.CustomImageNet(
        #     data_path=data_path,
        #     custom_grouping=[['NORMAL'], ['OPACITY']]
        # )
        # resume_path = 'robust_classifiers/rsna_classifier.pt'
        # self.classifier, _ = robustness.model_utils.make_and_restore_model(arch='resnet50',
        #                                                                    dataset=ds,
        #                                                                    resume_path=resume_path)
        # self.classifier.to(self.device)
        # self.classifier.eval()
        
        ###### MIMIC #########
        # data_path = '/home/wonjun29/data/mimic'
        ds = robustness.datasets.PNP_MIMIC('')
        # resume_path = 'robust_classifiers/mimic_red_classifier.pt'
        # resume_path = '/home/wonjun/repos/pnp/classifiers/red_aug_eps3_attacksteps20/199_checkpoint.pt'
        # resume_path = '/home/wonjun/repos/pnp/classifiers/red_aug_11710--_eps3_attacksteps20/199_checkpoint.pt'
        resume_path = '/home/wonjun29/code/pnp/classifiers/red_aug_11710--_eps3_attacksteps20/199_checkpoint.pt'
        self.classifier, _ = robustness.model_utils.make_and_restore_model(
            arch='resnet50',
            dataset=ds,
            resume_path=resume_path
        )
        self.classifier.to(self.device)
        self.classifier.eval()



    def _compute_probabilities(self, x, classifier):
        # logits = classifier(_map_img(x))
        logits, _ = classifier(_map_img(x))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return log_probs, probs

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep


    def perturb(self, x, y):
        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        print('shapes x', x.shape[-1], self.model_config["image_size"])
        if x.shape[-1] != self.model_config["image_size"]:
            x = transforms.Resize(self.image_size)(x)
            print('shapes x after', x.shape)
        self.init_image = (x.to(self.device).mul(2).sub(1).clone())

        def cond_fn_clean(x, t, y=None, eps=None):
            grad_out = torch.zeros_like(x)
            x = x.detach().requires_grad_()
            t = self.unscale_timestep(t)
            with torch.enable_grad():
                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )
                x_in = out["pred_xstart"]

            # compute classifier gradient
            keep_denoising_graph = self.args.denoise_dist_input
            with torch.no_grad():
                if self.args.classifier_lambda != 0:
                    with torch.enable_grad():
                        log_probs_1, probs_1 = self._compute_probabilities(x_in, self.classifier)
                        target_log_confs_1 = log_probs_1[range(1), y.view(-1)]
                        grad_class = torch.autograd.grad(target_log_confs_1.mean(), x,
                                                            retain_graph=keep_denoising_graph)[0]

                    if self.args.enforce_same_norms:
                        grad_, norm_ = _renormalize_gradient(grad_class, eps)
                        grad_class = self.args.classifier_lambda * grad_
                    else:
                        grad_class *= self.args.classifier_lambda

                    grad_out += grad_class

                # distance gradients
                if self.args.lp_custom: # and self.args.range_t < self.tensorboard_counter:
                    if not keep_denoising_graph:
                        diff = x_in - self.init_image
                        lp_grad = compute_lp_gradient(diff, self.args.lp_custom)
                    else:
                        with torch.enable_grad():
                            diff = x_in - self.init_image
                            lp_dist = compute_lp_dist(diff, self.args.lp_custom)
                            lp_grad = torch.autograd.grad(lp_dist.mean(), x)[0]
                    if self.args.quantile_cut != 0:
                        pass

                    if self.args.enforce_same_norms:
                        grad_, norm_ = _renormalize_gradient(lp_grad, eps)
                        lp_grad = self.args.lp_custom_value * grad_
                    else:
                        lp_grad *= self.args.lp_custom_value

                    grad_out -= lp_grad
                    
            return grad_out

        # gen_func = self.diffusion.p_sample_loop_progressive

        samples = self.diffusion.p_sample_loop_progressive(
            model = self.model,
            shape = (1, 3, 256, 256),
            clip_denoised=False,
            model_kwargs={
                "y": torch.tensor(y, device=self.device, dtype=torch.long)
                # torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
            },
            cond_fn=cond_fn_clean,
            progress=True,
            skip_timesteps=self.args.skip_timesteps,
            init_image=self.init_image,
            postprocess_fn=None,
            randomize_class=False,
            resizers=None,
            range_t=self.args.range_t,
            eps_project=self.args.eps_project,
            ilvr_multi=self.args.ilvr_multi,
            seed=self.args.seed
        )

        total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
        print('num total steps is', total_steps)
        
        for i, sample in enumerate(samples):
            print(i)
            
            # pred = self.classifier(sample['pred_xstart'])
            # pred_for_target_class = pred[0][y.item()].item()
                        
            fig, ax = plt.subplots(1, 1, figsize=(4,4))
            arr = np.transpose(sample['pred_xstart'][0].add(1).div(2).clamp(0,1).detach().cpu().numpy(), (1,2,0))
            ax.imshow(arr)
            # ax.set_title("target class pred: " + f"{pred_for_target_class:.4f}")
            plt.show()
            # fig.savefig(f'./reverse_diffusion_steps_images_nonrobust_dino/{i}.jpg')
            
            if i == total_steps:
                sample_final = sample
                
        return sample_final["pred_xstart"].add(1).div(2).clamp(0, 1)