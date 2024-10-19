import glob

from models.DeiT_utils.DeiT_robust_model import load_model_ext

try:
    import clip
    from PyTorch_CIFAR10.cifar10_models.resnet import resnet50
except Exception as err:
    print(str(err))
from torchvision.models import resnet50 as resnet50_in1000
from utils_svces.model_normalization import Cifar10Wrapper, ImageNetWrapper
from RATIO_utils.run_file_helpers import factory_dict
from RATIO_utils.run_file_helpers import models_dict as models_wrappers_dict
from models.Anon1s_smaller_radius_net import PreActResNet18 as PreActResNet18_Anon1
from models.BiTM import KNOWN_MODELS as BiT_KNOWN_MODELS
from models.SAMNets import WideResNet
#from models.BiTM_2 import ResNetV2
from models.preact_resnet import PreActResNet18
from functools import partial
import torch.nn as nn
import socket
#from robustness import model_utils, datasets

from robust_finetuning import utils as rft_utils
from torchvision.datasets.folder import DatasetFolder, has_file_allowed_extension,\
    pil_loader, accimage_loader, default_loader
from .MadryImageNet1000resnet import load_model as load_model_Madry

from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.model_zoo import model_dicts

from models.cifar_resnet import ResNet18
from models.ViT import VisionTransformer, CONFIGS
from PIL import Image
from collections import OrderedDict
from torchvision import transforms
try:
    from perceptual_advex.utilities import get_dataset_model
    from datasets import get_FundusKaggle
except Exception as err:
    print(str(err))
import RATIO_utils.datasets as dl
from torch.autograd import Variable
import os, sys
import shutil
import requests
import io
from .Wrappers import *

import numpy as np

__all__ = ['rm_substr_from_state_dict', 'Gowal2020UncoveringNet_load', 'PAT_load', 'RLAT_load', 'models_dict',
           'descr_args_generate', 'descr_args_rst_stab',
           'temperature_scaling_dl_dict', 'image_loader', 'ask_overwrite_folder', 'pretty',
           'loader_all', 'get_NVAE_MSE', 'BackgroundSubtractionWrapper', 'get_NVAE_class_model',
           'FIDDataset', 'Evaluator_FID_base_path', 'Evaluator_model_names_dict',
           'FeatureDist_implemented']

def insensitive_glob(pattern):
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob.glob(''.join(map(either, pattern)))

def get_base_dir(project_folder=None):

    if project_folder is not None:
        return project_folder
    else:
        raise ValueError('No project folder specified')

def get_base_dir_Anon1_finetuning(project_folder):
    machine_name = socket.gethostname()

    base = 'RATIO/robust_finetuning/trained_models/'

    return base


# To save RAM, for now do not use the FeatureDist
FeatureDist_implemented = lambda dataset_name: dataset_name in ['cifar10'] and False


def loader_all(is_CLIP_mode, device, model, kwargs, device_ids=None):
    if device_ids is None:
        return model(**kwargs) if is_CLIP_mode else model(**kwargs).to(device)
    else:
        return nn.DataParallel(model(**kwargs), device_ids=device_ids)


    #ImageNet1000ModelsPath = f'{get_base_dir()}/ACSM/ImageNet1000Models'

Evaluator_FID_base_path = 'ACSM/exp/logits_evolution_FID/image_samples'

Evaluator_model_names_cifar10 = ['benchmark-Gowal2020Uncovering_extra-L2', # 0, 200 |FID: 12.214471610707733, 80MTiny - FID:  73.48584942960701 (FID: 52.2703979513268)
           'BiT-M-R50x1_CIFAR10_nonrobust', #          1, 800 |FID: 31.798684438118755, 80MTiny - FID:  80.9394749163722 (FID:  76.00931802811118)
           'benchmark-Augustin2020Adversarial-L2', #'ResNet50', #                               2, 800 |FID: 7.682910516684672, 80MTiny -FID:  39.37918229989168(FID:  25.409702995609962), prior(CIFAR-10) - FID: 13.00980479007643, 80MTiny-prior - FID:  25.361933893836976
           'ResNet50_nonrobust', #                     3 |FID: 25.33756795783256, 80MTiny - FID:  82.75083332097341(FID:  68.20513836145074)
           'rst_stab', #                               4, 200 |FID:  5.037825277251386, 80MTiny - FID:  73.23352025516061(FID:  50.508438071267506)
           'benchmark-Gowal2020Uncovering-L2', #                    5 |FID:  8.239073025157722, 80MTiny - FID:  68.03733490306246
           'benchmark-Gowal2020Uncovering_improved', #           6, 200 |FID: 8.559810109140585, 80MTiny - FID:  56.911254301278234
           'benchmark-Wu2020Adversarial-L2', #                      7 |FID:  7.963537459704412, 80MTiny - FID:  67.38042324168163 (FID:  47.36846537530215), prior(CIFAR-10) - FID: 10.154340519585276
           'benchmark-PAT_improved', #                           8, 800 |FID:  9.070679516064502, 80MTiny - FID:  61.79206184439818 (FID:  45.75354326455374), 80MTiny-prior - FID:  45.465010488718406
           'benchmark-RLAT_improved', #                          9 |
           'benchmark-0.02l2:Anon1_small_radius_experimental', # 10, 800
           'benchmark-0.1l2:Anon1_small_radius_experimental',  # 11, 800
           'benchmark-0.25l2:Anon1_small_radius_experimental', # 12, 800
           'WideResNet34x10_feature_model',                      # 13
           'ViT-B_16_CIFAR10_nonrobust',                         # 14
           'benchmark-Hendrycks2020AugMix_ResNeXt-corruptions',  # 15, 800
           'benchmark-0.5l2:Anon1_small_radius_experimental',  # 16, 800
           'benchmark-12l1:Anon1_small_radius_experimental',   # 17, 800
           'benchmark-8,255linf:Anon1_small_radius_experimental', # 18, 800
           'benchmark-1l2:Anon1_small_radius_experimental',     # 19, 800
           'benchmark-0.75l2:Anon1_small_radius_experimental',  # 20, 800
           'benchmark-Augustin2020Adversarial_34_10_extra-L2',      # 21, 1000
           'benchmark-SAM_experimental',                            # 22, 200
           'benchmark-Gowal2020Uncovering_improved-L1.5',            # 23
           'benchmark-Augustin2020Adversarial_34_10_extra_improved-L1.5', # 24
           'benchmark-0.5l1.5:Anon1_small_radius_experimental',  # 25, 800
           'benchmark-1l1.5:Anon1_small_radius_experimental',  # 26, 800
           'benchmark-1.5l1.5:Anon1_small_radius_experimental',  # 27, 800
           'benchmark-2l1.5:Anon1_small_radius_experimental',  # 28, 800
           'benchmark-2.5l1.5:Anon1_small_radius_experimental']  # 29, 800

Evaluator_model_names_imagenet1000 = [
                                  'benchmark-Madry_l2_experimental', #    0 | FID
                                  'benchmark-Madry_linf_experimental', #      1 | FID
                                  'benchmark-Madry_l2_improved_ep_1', #       2 | FID
                                  'benchmark-Madry_l2_improved_ep_3', #       3 | FID
                                  'benchmark-Madry_l2_improved_eps_1', #      4 | FID
                                  'benchmark-Madry_linf_improved_ep_1', #     5 | FID
                                  'ResNet50IN1000_nonrobust',           #     6
                                  'benchmark-Madry_l2_improved_ep_1l1',   #     7, by this l1 fine-tuning is meant of the l2 robust model
                                  'benchmark-MicrosoftResNet50,experimental,l2,eps,0.25',       # 8
                                  'benchmark-MicrosoftResNet50,experimental,l2,eps,0.5',        # 9
                                  'benchmark-MicrosoftResNet50,experimental,l2,eps,1',          # 10
                                  'benchmark-MicrosoftResNet50,experimental,l2,eps,3',          # 11
                                  'benchmark-MicrosoftResNet50,experimental,l2,eps,5',          # 12
                                  'benchmark-MicrosoftWide_ResNet50_4,experimental,l2,eps,0.25',   # 13
                                  'benchmark-MicrosoftWide_ResNet50_4,experimental,l2,eps,0.5',     #14
                                  'benchmark-MicrosoftWide_ResNet50_4,experimental,l2,eps,1',       #15
                                  'benchmark-MicrosoftWide_ResNet50_4,experimental,l2,eps,3',       #16
                                  'benchmark-MicrosoftWide_ResNet50_4,experimental,l2,eps,5',       #17
                                  'benchmark-Madry_l2_improved_ep_1l1.5',                           #18
                                  'timm,tf_efficientnet_b7_ap,nonrobust',                           #19
                                  'timm,tf_efficientnet_b7_ns,nonrobust',                           #20
                                  'timm,swin_large_patch4_window12_384,nonrobust',                  #21
                                  'timm,tf_efficientnet_b7,nonrobust',                              #22
                                  'benchmark-DeiTrobust_experimental']                              #23
                                  #'BiT-M-R50x1IN1000_nonrobust']       #     6

Evaluator_model_names_funduskaggle = ['benchmark-Max:experimental,ResNet50,TRADES_04-04-2021_07:56:59',
                                      'benchmark-Anon1:finetuning_experimental,Wide_ResNet50_4,model_2021-09-23SPACE13:40:42.945427',
                                      'benchmark-Anon1:finetuning_experimental,Max=ResNet50,model_2021-09-29SPACE13:06:22.590333',
                                      'benchmark-Anon1:finetuning_experimental,Wide_ResNet50_4,model_2021-09-28SPACE19:52:12.481420',
                                      'benchmark-Max:experimental,ResNet50,TRADES_24-10-2021_22:55:07',
                                      'benchmark-Max:experimental,ResNet50,TRADES_02-11-2021_13:55:10',
                                      'benchmark-Max:experimental,ResNet50,TRADES_06-11-2021_15:39:13',
                                      'benchmark-Max:experimental,ResNet50,TRADES_01-10-2021_22:28:29',
                                      'benchmark-Max:experimental,ResNet50,plain_02-04-2021_08:15:08']

Evaluator_model_names_dict = {'cifar10': Evaluator_model_names_cifar10,
                              'imagenet1000': Evaluator_model_names_imagenet1000,
                              'tinyimages': Evaluator_model_names_cifar10,
                              'oa-imagenet': Evaluator_model_names_imagenet1000,
                              'funduskaggle': Evaluator_model_names_funduskaggle}



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def make_dataset(directories, extensions=None):
    instances = []
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for directory in directories:
        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, torch.Tensor([-1])
                    instances.append(item)
    return instances


# ToDo: Class index doesn't matter here, right?
class FIDDataset(DatasetFolder):

    def __init__(self, roots, transform=None):
        super(DatasetFolder, self).__init__(root=roots[0], transform=transform)

        self.samples = make_dataset(roots, IMG_EXTENSIONS)
        self.imgs = self.samples
        self.loader = default_loader

    def _find_classes(self, dir):
        classes = [None for d in os.scandir(dir) if d.name.endswith('last.png')]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


class ImageNetCustom(DatasetFolder):

    def __init__(self, root, train=False, download=True, transform=None, target_transform=None,
                 loader=default_loader, label_mapping=None):
        super(ImageNetCustom, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)#,
                                          #label_mapping=label_mapping)
        # ToDo: only works for testset! Write train/test cases!
        self.imgs = self.samples


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

"""
         'BiT-M-R50x1-CIFAR10_nonrobust', 
         'ResNet50', 
         'ResNet50_nonrobust',
         'rst_stab', 
         'benchmark-Gowal2020Uncovering', 
         'benchmark-Gowal2020Uncovering_improved', 
         'benchmark-Wu2020Adversarial', 
         'benchmark-PAT_improved', 
         'benchmark-RLAT_improved',
         """

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


# ToDo: put the cpkt call in a separate file
def Gowal2020UncoveringNet_load(threat_model='L2', project_folder=None):
    #model = Gowal2020UncoveringNet()
    # definition of the model is the same as Gowal2020Uncovering_extra, Gowal2020Uncovering_70_16_extra
    if threat_model == 'L1.5':
        model_path = f'{get_base_dir(project_folder)}/Cifar10Models/Gowal2020Uncovering_improved/' \
                     f'model_2021-10-21 15:47:59.655507 lr=0.01000 piecewise-5-5-5 ep=3 attack=afw fts=50 seed=0 iter=10 finetune_model at=L1.5 eps=1.5 balanced 500k no_rot/' \
                     f'ep_3.pth'
    elif threat_model == 'L2':
        model_path = f'{get_base_dir(project_folder)}/Cifar10Models/Gowal2020Uncovering_improved/' \
                                     f'model_2021-03-10 20:21:59.054345 lr=0.01000 piecewise-5-5-5 ep=3 ' \
                                     f'attack=apgd fts=50 seed=0 iter=10 finetune_model at=Linf L1 ' \
                                     f'balanced 500k no_rot/ep_3.pth'
    else:
        raise ValueError('Such norm is not supported.')

    model = model_dicts[BenchmarkDataset('cifar10')][ThreatModel('L2')]['Gowal2020Uncovering']['model']()
    model.load_state_dict(
        rm_substr_from_state_dict(
            torch.load(model_path, map_location=torch.device('cpu')), 'module.')
        , strict=True)

    return model


def Augustin2020Adversarial_34_10Net_load(threat_model='L1.5'):

    if threat_model == 'L1.5':
        model_path = f'{get_base_dir()}/Cifar10Models/' \
                     f'Augustin2020Adversarial_34_10_extra/' \
                     f'ratio_finetuned_l15_asam.pth'
    else:
        raise ValueError('Such norm is not supported.')

    # ToDo: Sehwag2021Proxy uses the same model as Augustin2020Adversarial_34_10_extra, but w/o a wrapper in the definition
    model = model_dicts[BenchmarkDataset('cifar10')][ThreatModel('L2')]['Sehwag2021Proxy']['model']()
    model.load_state_dict(
        rm_substr_from_state_dict(
            torch.load(model_path, map_location=torch.device('cpu')), 'module.')
        , strict=True)

    model = Cifar10Wrapper(model)

    return model


def MicrosoftNet(model_arch,
                 norm,
                 epsilon):

    if 'Wide_ResNet50_4' in model_arch:
        path_model = f'{get_base_dir()}/robust_finetuning/external_models/wide_resnet50_4_{norm}_eps{epsilon}.ckpt'
    elif 'ResNet50' in model_arch:
        path_model = f'{get_base_dir()}/ImageNet1000Models/microsoft/resnet50_{norm}_eps{epsilon}.ckpt'
    else:
        raise ValueError(f'Model arch {model_arch} is not implemented here.')

    model, checkpoint = model_utils.make_and_restore_model(
        arch=model_arch.lower(), dataset=datasets.ImageNet(''), resume_path=path_model)

    return model

def MadryNet(norm,
             improved,
             num_pretrained_epochs,
             epsilon_finetuned=None,
             project_folder=None
             ):



    model_paths_dict = {
                        'l2_improved_1l1.5_ep': 'Madry_l2_improved/model_2021-10-21 20:21:20.122644 imagenet lr=0.01000 piecewise-5-5-5 ep=1 wd=0.0001 attack=afw fts=2 seed=0 iter=10 finetune_model at=L1.5 eps=12.5 balanced no_rot/ep_1.pth',
                        'l2_improved_1l1_ep': 'Madry_l2_improved/model_2021-03-17 09:25:30.985477 imagenet lr=0.01000 piecewise-5-5-5 ep=1 wd=0.0001 attack=apgd fts=2 seed=0 iter=15 finetune_model at=L1 balanced no_rot/ep_1.pth',
                        'l2_improved_1_ep': 'Madry_l2_improved/model_2021-03-16 11:38:45.988619 imagenet lr=0.01000 piecewise-5-5-5 ep=1 wd=0.0001 attack=apgd fts=2 seed=0 iter=5 15 finetune_model at=Linf L1 balanced no_rot/ep_1.pth',
                        'l2_improved_3_ep': 'Madry_l2_improved/model_2021-05-02 15:03:55.621750 imagenet lr=0.01000 piecewise-5-5-5 ep=3 wd=0.0001 attack=apgd fts=2 seed=0 iter=5 15 finetune_model at=Linf L1 balanced no_rot/ep_3.pth',
                        'linf_improved_1_ep': 'Madry_linf_improved/model_2021-03-15 13:03:18.873067 imagenet lr=0.01000 piecewise-5-5-5 ep=1 wd=0.0001 attack=apgd fts=1 seed=0 iter=5 15 finetune_model at=Linf L1 balanced no_rot/ep_1.pth',
                        }

    ImageNet1000ModelsPath = f'{get_base_dir(project_folder)}/ImageNet1000Models'

    model = load_model_Madry(modelname='Engstrom2019Robustness',
                             norm=norm,
                             modeldir=ImageNet1000ModelsPath+'/Madry')
    # ToDo: do checkpoint loading only once to save time
    if improved:
        if epsilon_finetuned is not None:
            model_path = os.path.join(ImageNet1000ModelsPath,
                                      model_paths_dict[norm + f'_improved_{str(epsilon_finetuned)}_eps'])
        else:
            model_path = os.path.join(ImageNet1000ModelsPath,
                                      model_paths_dict[norm+f'_improved_{str(num_pretrained_epochs)}_ep'])
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=True)

    return model

def Anon1s_smaller_radius(eps, project_folder=None):
    model_paths_dict = {
                        '0.25l2': 'model_2021-05-06 18:29:13.639852 lr=0.05000 piecewise-5-5-5 ep=3 attack=apgd act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L2 eps=.25 balanced no_rot continued [model_2021-03-15 14:05:42.938809 ep_72]/ep_3.pth',
                        '0.1l2': 'model_2021-05-06 19:57:35.842192 lr=0.05000 piecewise-5-5-5 ep=3 attack=apgd act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L2 eps=.1 balanced no_rot continued [model_2021-03-15 14:05:42.938809 ep_72]/ep_3.pth',
                        '0.02l2': 'model_2021-05-06 20:21:58.358943 lr=0.05000 piecewise-5-5-5 ep=3 attack=apgd act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L2 eps=.02 balanced no_rot continued [model_2021-03-15 14:05:42.938809 ep_72]/ep_3.pth',
                        '0.5l2': 'l2_0.5/pretr_L2.pth',
                        '0.75l2': 'model_2021-09-23 16:42:33.694120 lr=0.05000 superconverge ep=30 attack=apgd act=softplus1 fts=rand seed=0 iter=5 finetune_model at=L2 eps=.75 balanced no_wd4bn no_rot/ep_30.pth',
                        '1l2': 'model_2021-09-23 12:39:26.072195 lr=0.05000 superconverge ep=30 attack=apgd act=softplus1 fts=rand seed=0 iter=5 finetune_model at=L2 eps=1. balanced no_wd4bn no_rot/ep_30.pth',
                        '12l1': 'l1_12/pretr_L1.pth',
                        '8,255linf': 'linf_8_255/pretr_Linf.pth',
                        # l1.5 finetuned
                        '0.5l1.5': f'{get_base_dir(project_folder)}/ImageNet1000Models/Anon1_smaller_radius/model_2021-10-31 13:50:41.062428 lr=0.05000 superconverge ep=30 attack=afw act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L1.5 eps=.5 balanced no_wd4bn no_rot/ep_30.pth',
                        '1l1.5': f'{get_base_dir(project_folder)}/ImageNet1000Models/Anon1_smaller_radius/model_2021-10-30 14:25:23.725710 lr=0.05000 superconverge ep=30 attack=afw act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L1.5 eps=1 balanced no_wd4bn no_rot/ep_30.pth',
                        '1.5l1.5': f'{get_base_dir(project_folder)}/ImageNet1000Models/Anon1_smaller_radius/model_2021-10-20 17:56:14.827435 lr=0.05000 superconverge ep=30 attack=afw act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L1.5 eps=1.5 balanced no_wd4bn no_rot/ep_30.pth',
                        '2l1.5': f'{get_base_dir(project_folder)}/ImageNet1000Models/Anon1_smaller_radius/model_2021-10-30 14:25:23.735285 lr=0.05000 superconverge ep=30 attack=afw act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L1.5 eps=2. balanced no_wd4bn no_rot/ep_30.pth',
                        '2.5l1.5': f'{get_base_dir(project_folder)}/ImageNet1000Models/Anon1_smaller_radius/model_2021-10-30 14:25:58.906229 lr=0.05000 superconverge ep=30 attack=afw act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L1.5 eps=2.5 balanced no_wd4bn no_rot/ep_30.pth'
    }

    model = PreActResNet18_Anon1(n_cls=10, activation='softplus1')
    model_path = os.path.join(get_base_dir(project_folder), 'Anon1_smaller_radius', model_paths_dict[eps])
    state_dict = torch.load(model_path)
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'], strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)

    return model


# ToDo: put the cpkt call in a separate file
def PAT_load(arch='resnet50',
             project_folder=None):
    model_path = f'{get_base_dir(project_folder)}/Cifar10Models/PAT/cifar/pat_self_0.5.pt',
    #This works for resnet50
    _, model = get_dataset_model(
        dataset='cifar',
        arch=arch,
        checkpoint_fname=model_path,
    )

    #model = getattr(torchvision_models, arch)(pretrained=pretrained)
    #state = torch.load(model_path)
    #model.load_state_dict(state['model'])


    #model = AlexNetFeatureModel(model)
    #model.load_state_dict(torch.load(model_path)['model'])

    return model

def RLAT_load(project_folder=None):
    model_path = f'{get_base_dir(project_folder)}/Cifar10Models/RLAT/rlat-eps=0.05-augmix-cifar10/rlat-eps=0.05-augmix-cifar10.pt'
    model = PreActResNet18(n_cls=10)
    model.load_state_dict(torch.load(model_path)['last'])
    model.eval()
    return model


def BiTM_get_weights(bit_variant):
  response = requests.get(f'https://storage.googleapis.com/bit_models/{bit_variant}.npz')
  response.raise_for_status()
  return np.load(io.BytesIO(response.content))


def BiTM_load(model_name='', class_labels=None, dataset='cifar10'):
    #weights_cifar10 = get_weights(model_name)
    #model = ResNetV2(ResNetV2.BLOCK_UNITS['r50'], width_factor=1, head_size=10)  # NOTE: No new head.
    #model.load_from(weights_cifar10)
    #model.eval()
    print('BiT model name is', model_name, 'dataset is', dataset)
    print('head_size is', len(class_labels))
    #model_name = 'BiT-M-R50x1'
    ##model = ResNetV2(ResNetV2.BLOCK_UNITS['r50'], width_factor=1, head_size=10)
    model = BiT_KNOWN_MODELS[model_name](head_size=len(class_labels), zero_head=True)
    # ToDo: only BiT-M-R50x1 is supported currently
    print('model is', model)

    model.load_from(np.load(f"{get_base_dir()}/BiT-pytorch/big_transfer/BiT-M-R50x1.npz")) ## (BiTM_get_weights('BiT-M-R50x1-CIFAR10'))
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(f"{get_base_dir()}/BiT-pytorch/big_transfer/output/{dataset}/bit.pth.tar", map_location="cpu")
    print('checkpoint', f"{get_base_dir()}/BiT-pytorch/big_transfer/output/{dataset}/bit.pth.tar")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def ViT_load(model_name='', class_labels=None, img_size=224, device='cuda:0'):
    config = CONFIGS['ViT-B_16']

    model = VisionTransformer(config, img_size=img_size, zero_head=True, num_classes=len(class_labels))
    model.load_from(np.load(f"{get_base_dir()}/VIT-pytorch/ViT-pytorch/checkpoint/ViT-B_16.npz"))
    checkpoint = torch.load(f"{get_base_dir()}/VIT-pytorch/ViT-pytorch/output/cifar10-100_500_checkpoint.bin",
                            map_location="cpu")
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    model_ = lambda x: model(x)[0]

    return model_

def CLIP_model(model_name='', device=None, class_labels=None):
    model_, preprocess = clip.load(name=model_name, device=device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_labels]).to(device)
    text_features = model_.encode_text(text_inputs).type(torch.float32)
    text_features /= text_features.norm(dim=1, keepdim=True)
    model = partial(CLIP_model_, model=model_, text_features=text_features, preprocess=preprocess)
    return model

def CLIP_model_(image, text_features, model, preprocess):
    #image_input = preprocess(image)
    image_features = model.encode_image(image).type(torch.float32)
    #print('norm', image_features.norm(dim=1, keepdim=True))
    image_features = image_features / (image_features.norm(dim=1, keepdim=True))
    logits = 100 * image_features @ text_features.T
    #similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    #values, indices = similarity[0].topk(10)
    return logits


def SAMNets(device='cuda:0'):
    depth, width_factor, dropout = 16, 8, 0.0
    dataset = 'cifar10'
    m_path = f'{get_base_dir()}/SAM_pytorch/sam/example/trained_models/model_2021-10-19 16:37:26.596415/ep_200.pth'
    model = WideResNet(depth, width_factor, dropout, in_channels=3, labels=10).to(device)
    state_dict = torch.load(m_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = models_wrappers_dict[dataset](model)
    model.eval()
    return model


def MaxNets(dataset_name, arch, model_name_id, num_classes, img_size, device='cuda:0', project_folder=''):
    if img_size in factory_dict:
        model, model_name, config = factory_dict[img_size].build_model(arch, num_classes)
    else:
        raise ValueError(f'Model factories are supported only for image sizes {factory_dict.keys()},'
                         f' and not for {img_size}!')

    print(f'searching in {os.path.join(get_base_dir(project_folder), "fundusKaggleModels", f"*{model_name_id}*/best.pth")}')
    state_dict_file = insensitive_glob(os.path.join(get_base_dir(project_folder), "fundusKaggleModels", f"*{model_name_id}*/best.pth"))[0]
    print(f'resotring file from {state_dict_file}')

    state_dict = torch.load(state_dict_file, map_location=device)
    model.load_state_dict(state_dict)
    model = models_wrappers_dict[dataset_name.lower()](model)

    return model


def DeiTNets(model_name='deit_small_patch16_224_adv',
             model_path='ImageNet1000Models/DeiT/model_2021-12-16 16:48:41.667201 imagenet lr=0.01000 piecewise-5-5-5 ep=1 wd=0.0001 attack=apgd fts=3 seed=0 iter=5 15 finetune_model at=Linf L1 balanced no_rot/ep_1.pth',
             project_folder=None):

    model = load_model_ext(model_name)
    ckpt_path = os.path.join(project_folder, model_path)
    a = torch.load(ckpt_path)
    print(dir(a))
    model.load_state_dict(a)

    return model


def Anon1FinetuningNets(dataset_name, arch, model_name_id, num_classes, additional_hidden=0, project_folder=None):
    # ToDo: do we even need this resume path?
    arch = arch.lower()
    resume_path = f'{get_base_dir(project_folder)}/RATIO/robust_finetuning/external_models/wide_resnet50_4_l2_eps1.ckpt'
    if 'max=' in arch:
        arch = arch.replace('max=', '')
        # ToDo: use variable for img_size
        img_size = 224
        if img_size in factory_dict:
            model, model_name, config = factory_dict[img_size].build_model(arch, num_classes)
        else:
            raise ValueError(f'Model factories are supported only for image sizes {factory_dict.keys()},'
                             f' and not for {img_size}!')

        state_dict_file = \
            insensitive_glob(os.path.join(get_base_dir_Anon1_finetuning(),
                                          f'{model_name_id.replace("SPACE", " ")}*/ep_9.pth'))[0]

        print(f'resotring file from {state_dict_file}')
        state_dict = torch.load(state_dict_file, map_location='cpu')
        model.load_state_dict(state_dict)
        model = models_wrappers_dict[dataset_name.lower()](model)

    else:
        print(f'[Replacing the last layer with {additional_hidden} '
              f'hidden layers and 1 classification layer that fits the {dataset_name} dataset.]')

        model, checkpoint = model_utils.make_and_restore_model(arch=arch, dataset=datasets.ImageNet(''), resume_path=resume_path)

        while hasattr(model, 'model'):
            model = model.model

        model = rft_utils.ft(
            arch, model, num_classes, additional_hidden)

        model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=datasets.ImageNet(''),
                                                               add_custom_forward=False)
        state_dict_file = insensitive_glob(os.path.join(get_base_dir_Anon1_finetuning(),
                                                        f'{model_name_id.replace("SPACE", " ")}*/ep_26.pth'))[0]  # 36

        state_dict = torch.load(state_dict_file)
        model.load_state_dict(state_dict)

        #model.cuda()
        #model.eval()
    return model

BiTMWrapper_mean = torch.tensor([0.5, 0.5, 0.5])
BiTMWrapper_std = torch.tensor([0.5, 0.5, 0.5])

BiTMCIFAR10Wrapper_size = 128
BiTMIN1000Wrapper_size = 480

try:
    models_dict = {
               # CIFAR10 Models
               'ResNet50': lambda **kwargs: Cifar10Wrapper(resnet50(**kwargs)),
               'ResNet50IN1000': lambda **kwargs: ImageNetWrapper(resnet50_in1000(**kwargs)),
               'BiT-M-R50x1': lambda **kwargs: BiTMWrapper(BiTM_load(**kwargs),
                                                              mean=BiTMWrapper_mean,
                                                              std=BiTMWrapper_std,
                                                              size=BiTMCIFAR10Wrapper_size),
               """
               'BiT-M-R50x1IN1000': lambda **kwargs: BiTMWrapper(BiTM_load(**kwargs),
                                                                 mean=BiTMWrapper_mean,
                                                                 std=BiTMWrapper_std,
                                                                 size=BiTMIN1000Wrapper_size),
               """
               
               'ViT-B': lambda **kwargs: BiTMWrapper(ViT_load(**kwargs),
                                                                mean=BiTMWrapper_mean,
                                                                std=BiTMWrapper_std,
                                                                size=None),
               'Gowal2020Uncovering_improved': Gowal2020UncoveringNet_load,
               'Augustin2020Adversarial_34_10_extra_improved': Augustin2020Adversarial_34_10Net_load,
               'PAT_improved': PAT_load,
               'RLAT_improved': RLAT_load,
               #'ResNet18_finetuned_ep15_improved': ResNet18_finetuned_load,
               'CLIP': CLIP_model,
               # ImageNet1000 Models
               'Madry': MadryNet,
               'Anon1_small_radius_experimental': Anon1s_smaller_radius,
               'Microsoft': MicrosoftNet,
               'Max': MaxNets,
               'Anon1:finetuning': Anon1FinetuningNets,
               'SAM': SAMNets,
               'DeiTrobust_experimental': DeiTNets
               }

except Exception as err:
    print(str(err))

descr_args_rst_stab = lambda project_folder : {'path': f'{get_base_dir(project_folder)}/Cifar10Models/RST_stab/AdvACET_24-02-2020_14:41:39/'
                               'cifar10_rst_stab.pt.ckpt',
                       'model': 'wrn-28-10'}


def descr_args_generate(threat_model=None, pretrained=False,
                        is_experimental=False, dataset_='cifar10', model_name=None, project_folder=None):
    if is_experimental:
        if threat_model is not None:
            return {'threat_model': threat_model, 'project_folder': project_folder}
        elif pretrained:
            print('using pretrained model')
            return {'pretrained': True}
        else:
            return {'project_folder': project_folder}
    else:
        return {
            'model_name': model_name,
            'dataset': dataset_,
            'threat_model': threat_model,
            'project_folder': project_folder
        }

# ToDo: generalize for different BiT models
temperature_scaling_dl_dict = lambda batch_size, img_size, project_folder, data_folder, model_name=None: \
    {#'cifar10': dl.get_CIFAR10_1(batch_size=batch_size, size=img_size), # ToDo: was the temperature computed correctly on BiT?
     'imagenet1000': dl.get_ImageNet1000_idx(idx_path=f'{get_base_dir(project_folder)}/ImageNet1000/imagenet_val_random_idx_calibration.npy', model_name=model_name,
                                             batch_size=batch_size, project_folder=project_folder, data_folder=data_folder),

     #'restrictedimagenet': dl.imagenet_subsets.get_restrictedImageNet(train=False, batch_size=batch_size,
     #                                                                 augm_type='none', num_samples=2000, balanced=False), # num_samples=200
     #'imagenet1000': '',
     #'funduskaggle': get_FundusKaggle(split='val', batch_size=batch_size, augm_type='none', size=img_size,
     #               # ToDo: put in the config
     #             background_subtraction=True, balanced=False, project_folder=project_folder, data_folder=data_folder)
     }
    #'lsun_scenes': 'LSUNScenesModels',
    #'celeba': 'CELEBAModels'}

loader = lambda imsize: transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

def tensor_loader(img_filepath, imsize, img_id):
    tensor_filepath = '/'.join(img_filepath.split('/')[:-1])
    tensor_filepath = glob.glob(os.path.join(tensor_filepath, '*.pt'), recursive=True)
    assert len(tensor_filepath) == 1, 'Only one tensor is expected to be in the folder.'
    tensor_ = torch.load(tensor_filepath)

    return tensor_[img_id][:, :, -imsize:].unsqueeze(0).cuda()

def image_loader(image_name, imsize):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(imsize)(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

def ask_overwrite_folder(folder, no_interactions, fatal=True, FID_calculation=False):
    if not os.path.exists(folder):
        os.makedirs(folder)
    elif no_interactions and not FID_calculation:
        shutil.rmtree(folder)
        os.makedirs(folder)
    elif not FID_calculation:
        response = input(f"Folder '{folder}' already exists. Overwrite? (Y/N)")
        if response.upper() == 'Y':
            shutil.rmtree(folder)
            os.makedirs(folder)
        elif fatal:
            print("Output image folder exists. Program halted.")
            sys.exit(0)
    else:
        pass

def get_weights(bit_variant):
  response = requests.get(f'https://storage.googleapis.com/bit_models/{bit_variant}.npz')
  response.raise_for_status()
  return np.load(io.BytesIO(response.content))


def get_NVAE_class_model(models_dict, class_id):
    if class_id in models_dict:
        return models_dict[class_id]
    else:
        return None

def get_NVAE_MSE(image, model, batch_size):
    if model is not None:
        image_out = model(image)
        output = model.decoder_output(image_out[0])
        output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
            else output.sample()
        return (output_img-image).view(batch_size, -1).norm(p=2, dim=1)
    else:
        return 'NA'





