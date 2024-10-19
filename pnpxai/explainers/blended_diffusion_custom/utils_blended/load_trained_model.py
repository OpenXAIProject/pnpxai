import torch
import os

try:
    from RATIO_utils.models.models_32x32.resnet import ResNet18, ResNet34, ResNet50
    from RATIO_utils.models.models_32x32.wide_resnet import WideResNet28x10, WideResNet34x20
    from RATIO_utils.models.models_32x32.wideresnet_carmon import WideResNet as WideResNetCarmon
    from RATIO_utils.models.models_32x32.fixup_resnet import fixup_resnet56
    from RATIO_utils.models.models_32x32.shake_pyramidnet import ShakePyramidNet
    from RATIO_utils.models.model_factory_32 import build_model as build_model32
    from RATIO_utils.models.model_factory_224 import build_model as build_model224
    from torchvision import models
    import torch.nn as nn

    from RATIO_utils.models.models_224x224.resnet_224 import resnet50 as imagenet_resnet50
    from RATIO_utils.models.models_224x224.efficientnet import EfficientNet
    from RATIO_utils.model_normalization import Cifar10Wrapper, Cifar100Wrapper, SVHNWrapper,\
    ImageNetWrapper, RestrictedImageNetWrapper, BigTransferWrapper
    from RATIO_utils.temperature_wrapper import TemperatureWrapper
    import RATIO_utils.models.ebm_wrn as wrn
    from RATIO_utils.models.big_transfer_factory import build_model_big_transfer
    from datasets import get_base_dir
except Exception as err:
    print(str(err))



def load_non_native_model(type, folder, device):
    # if type == 'EBM':
    #     f = wrn.CCF(depth=28, width=10, norm=None)
    #
    #     file = 'Cifar10Models/EBM/CIFAR10_MODEL.pt'
    #     state_dict = torch.load(file)
    #     f.load_state_dict(state_dict["model_state_dict"])
    #
    #     # n_steps_refine is number of pre sampling steps
    #     f = wrn.DummyModel(f, n_steps_refine=0)
    #
    #     # Use this line if you want to be able to run an adaptive attack
    #     # only matters if n_steps_refine>0
    #     # might blow up your memory though...
    #     f.detach = False
    #     density_model = f.to(device)
    #     density_model.eval()
    #
    #     density_model = wrn.gradient_attack_wrapper(density_model)
    # elif type == 'MadryCifarResNet50':
    #     from robustness import model_utils, datasets
    #
    #     DATA = 'CIFAR'
    #     DATA_SHAPE = 32
    #     DATA_PATH_DICT = {
    #         'CIFAR': '../ref_data'
    #     }
    #
    #     dataset_function = getattr(datasets, DATA)
    #     dataset = dataset_function(DATA_PATH_DICT[DATA])
    #
    #     model_kwargs = {
    #         'arch': 'resnet50',
    #         'dataset': dataset,
    #         'resume_path': f'Cifar10Models/MadryModels/ResNet50/{folder}.pt',
    #         'parallel' : False
    #     }
    #
    #     class MadryWrapper(torch.nn.Module):
    #         def __init__(self, density_model, normalizer):
    #             super().__init__()
    #             self.density_model = density_model
    #             self.normalizer = normalizer
    #
    #         def forward(self, img):
    #             normalized_inp = self.normalizer(img)
    #             output = self.density_model(normalized_inp, with_latent=False,
    #                                 fake_relu=False, no_relu=False)
    #             return output
    #
    #     model_madry, _ = model_RATIO_utils.make_and_restore_model(**model_kwargs)
    #
    #     density_model = MadryWrapper(model_madry.density_model, model_madry.normalizer)
    #     density_model.to(device)
    #     density_model.eval()
    # elif type == 'MadryRestrictedImageNet50':
    #     from robustness import model_utils, datasets
    #     dataset = datasets.DATASETS['restricted_imagenet']('/home/scratch/datasets/ImageNet/')
    #     model_kwargs = {
    #         'arch': 'resnet50',
    #         'dataset': dataset,
    #         'resume_path': f'RestrictedImageNetModels/MadryResNet50/{folder}.pt',
    #         'parallel' : False
    #     }
    #
    #     class MadryWrapper(torch.nn.Module):
    #         def __init__(self, density_model, normalizer):
    #             super().__init__()
    #             self.density_model = density_model
    #             self.normalizer = normalizer
    #
    #         def forward(self, img):
    #             normalized_inp = self.normalizer(img)
    #             output = self.density_model(normalized_inp, with_latent=False,
    #                                 fake_relu=False, no_relu=False)
    #             return output
    #
    #     model_madry, _ = model_RATIO_utils.make_and_restore_model(**model_kwargs)
    #
    #     density_model = MadryWrapper(model_madry.density_model, model_madry.normalizer)
    #     density_model.to(device)
    #     density_model.eval()
    if type == 'TRADESReference':
        model = ResNet50(num_classes=10)
        state_dict_file = f'{folder}.pt'
        state_dict = torch.load(state_dict_file, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    if type == 'Carmon':
        model = WideResNetCarmon(num_classes=10, depth=28, widen_factor=10)
        state_dict_file = f'Cifar10Models/{folder}.pt'
        checkpoint = torch.load(state_dict_file, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint )
        num_classes = checkpoint.get('num_classes', 10)
        normalize_input = checkpoint.get('normalize_input', False)
        def strip_data_parallel(s):
            if s.startswith('module'):
                return s[len('module.'):]
            else:
                return s
        state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    else:
        raise ValueError('Type not supported')

    return model


def get_root_folder():
    return os.path.dirname(os.path.abspath(__file__))

def get_filename(folder, architecture_folder, checkpoint, load_temp):
    if load_temp:
        load_folder_name = f'_temp_{folder}'
    else:
        load_folder_name = f'{folder}'

    if not checkpoint.isnumeric():
        state_dict_file = f'{get_root_folder()}/../{architecture_folder}/{load_folder_name}/{checkpoint}.pth'
    else:
        epoch = int(checkpoint)

        state_dict_file = f'{get_base_dir()}/robust_finetuning/trained_models/model_2021-10-04 09:38:36.343666 cifar10 lr=0.01000 piecewise-ft ep=10 attack=AdvACET fts=RB:Augustin2020Adversarial:L2 seed=0 at=L1 Linf eps=default iter=15' \
                          f'/ep_{epoch}.pth'

    return state_dict_file


non_native_model = ['EBM', 'Madry50', 'TRADESReference', 'MadryRestrictedImageNet50', 'Carmon']

def load_cifar_family_model(type, folder, checkpoint, device, dataset_dir, num_classes, load_temp=False, model_params=None):
    model, model_folder_post, _ = build_model32(type, num_classes, model_params=model_params)
    state_dict_file = get_filename(folder, os.path.join(dataset_dir, model_folder_post), checkpoint, load_temp)
    print('Loading CIFAR10 model from', state_dict_file)
    state_dict = torch.load(state_dict_file, map_location=device)
    for key_to_del in ['mu', 'sigma']:
        if key_to_del in state_dict:
            del state_dict[key_to_del]
    model.load_state_dict(state_dict)
    return model

def load_big_transfer_model(type, folder, checkpoint, device, dataset_dir, num_classes, load_temp=False, model_params=None):
    model, model_folder_post = build_model_big_transfer(type, num_classes)
    state_dict_file = get_filename(folder, os.path.join(dataset_dir, model_folder_post), checkpoint, load_temp)
    state_dict = torch.load(state_dict_file, map_location=device)
    model.load_state_dict(state_dict)
    return model

def load_imagenet_family_model(type, folder, checkpoint, device, dataset_dir, num_classes, load_temp=False, model_params=None):
    if type == 'efficientnet-b0':
        model = EfficientNet.from_name('efficientnet-b0', override_params={'num_classes': num_classes})
        state_dict_file = get_filename(folder, f'{dataset_dir}//EfficientNet-B0', checkpoint, load_temp)
        state_dict = torch.load(state_dict_file, map_location=device)
        model.load_state_dict(state_dict)
    # elif type == 'ResNet50':
    #     #density_model = imagenet_resnet50(pretrained=False, num_classes=num_classes)
    #     model = models.resnet50(pretrained=False)
    #     num_ftrs = model.fc.in_features
    #     model.fc = nn.Linear(num_ftrs, num_classes)
    #     state_dict_file = get_filename(folder, f'{dataset_dir}//ResNet50', checkpoint, load_temp)
    #     state_dict = torch.load(state_dict_file, map_location=device)
    #     model.load_state_dict(state_dict)
    # elif type == 'ResNet50':
    #     model = imagenet_resnet50(pretrained=False, num_classes=num_classes)
    #     state_dict_file = get_filename(folder, f'{dataset_dir}//ResNet50', checkpoint, load_temp)
    #     state_dict = torch.load(state_dict_file, map_location=device)
    #     model.load_state_dict(state_dict)
    else:
        model, model_folder_post, _ = build_model224(type, num_classes, **model_params)
        state_dict_file = get_filename(folder, f'{dataset_dir}/{model_folder_post}', checkpoint, load_temp)
        state_dict = torch.load(state_dict_file, map_location=device)
        model.load_state_dict(state_dict)

    return model

def load_model(type, folder, checkpoint, temperature, device, dataset='cifar10', load_temp=False,  model_params=None):
    dataset = dataset.lower()
    checkpoint = str(checkpoint)

    if dataset == 'cifar10':
        dataset_dir = 'Cifar10Models'
        num_classes = 10
        model_family = 'Cifar32'
    elif dataset == 'cifar100':
        dataset_dir = 'Cifar100Models'
        num_classes = 100
        model_family = 'Cifar32'
    elif dataset == 'svhn':
        dataset_dir = 'SVHNModels'
        num_classes = 10
        model_family = 'Cifar32'
    elif dataset == 'tinyImageNet':
        dataset_dir = 'TinyImageNetModels'
        num_classes = 200
        model_family = 'ImageNet224'
    elif dataset == 'restrictedimagenet':
        #dataset_dir = 'RestrictedImageNetModels'
        dataset_dir = 'RestrictedImageNetModels'
        num_classes = 9
        model_family = 'ImageNet224'
    elif dataset == 'imagenet':
        dataset_dir = 'ImageNetModels'
        num_classes = 1000
        model_family = 'ImageNet224'
    elif dataset == 'imagenet100':
        dataset_dir = 'ImageNet100Models'
        num_classes = 100
        model_family = 'ImageNet224'
    elif dataset == 'pets':
        dataset_dir = 'PetsModels'
        num_classes = 37
        model_family = 'ImageNet224'
    elif dataset == 'flowers':
        dataset_dir = 'FlowersModels'
        num_classes = 102
        model_family = 'ImageNet224'
    elif dataset == 'cars':
        dataset_dir = 'CarsModels'
        num_classes = 196
        model_family = 'ImageNet224'
    elif dataset == 'food-101':
        dataset_dir = 'Food-101Models'
        num_classes = 101
        model_family = 'ImageNet224'
    elif dataset == 'lsun_scenes':
        dataset_dir = 'LSUNScenesModels'
        num_classes = 10
        model_family = 'ImageNet224'
    else:
        raise ValueError('Dataset not supported')

    if type in non_native_model:
        model = load_non_native_model(type, folder, device)
        if temperature is not None:
            model = TemperatureWrapper(model, temperature)
        return model

    if 'BiT' in type:
        model = load_big_transfer_model(type, folder, checkpoint, device, dataset_dir, num_classes, load_temp=load_temp)
        model = BigTransferWrapper(model)
    else:
        if model_family == 'Cifar32':
            model = load_cifar_family_model(type, folder, checkpoint, device, dataset_dir, num_classes,
                                            load_temp=load_temp, model_params=model_params)
        elif model_family == 'ImageNet224':
            model = load_imagenet_family_model(type, folder, checkpoint, device, dataset_dir, num_classes,
                                               load_temp=load_temp, model_params=model_params)
        else:
            raise ValueError()

        if dataset == 'cifar10':
            model = Cifar10Wrapper(model)
        elif dataset == 'cifar100':
            model = Cifar100Wrapper(model)
        elif dataset == 'svhn':
            model = SVHNWrapper(model)
        elif dataset == 'tinyimagenet':
            model = Cifar100Wrapper(model)
        elif dataset == 'imagenet':
            model = ImageNetWrapper(model)
        elif dataset == 'restrictedimagenet':
            model = RestrictedImageNetWrapper(model)
        elif dataset == 'imagenet100':
            model = ImageNetWrapper(model)
        elif dataset == 'pets':
            model = ImageNetWrapper(model)
        elif dataset == 'food-101':
            model = ImageNetWrapper(model)
        elif dataset == 'cars':
            model = ImageNetWrapper(model)
        elif dataset == 'flowers':
            model = ImageNetWrapper(model)
        elif dataset == 'lsun_scenes':
            model = ImageNetWrapper(model)
        else:
            raise ValueError('Dataset not supported')

    model.to(device)

    if temperature is not None:
        model = TemperatureWrapper(model, temperature)

    model.eval()
    return model