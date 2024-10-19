import socket

import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models as torch_models
from typing import Tuple
from torch import Tensor
import torchvision


def resnet50_return_layers(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_return_layers('resnet50', torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


model_class_dict = {'pt_vgg': torch_models.vgg16_bn,
                    'pt_resnet': torch_models.resnet50,
                    'pt_resnet_return_layers': resnet50_return_layers,
                    'pt_inception': torch_models.inception_v3,
                    'pt_densenet': torch_models.densenet121}


def _resnet_return_layers(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_return_layers(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(torchvision.models.resnet.model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class ResNet_return_layers(torchvision.models.ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual,
                 groups=groups, width_per_group=width_per_group, replace_stride_with_dilation=replace_stride_with_dilation,
                 norm_layer=norm_layer)

        self._return_layers = False

    @property
    def return_layers(self):
        """I'm the 'x' property."""
        print("getter of x called")
        return self._return_layers

    @return_layers.setter
    def return_layers(self, value):
        print("setter of x called")
        self._return_layers = value

    @return_layers.deleter
    def return_layers(self):
        print("deleter of x called")
        del self._return_layers

    def forward(self, x: Tensor) -> Tensor:
        print('return layers', self._return_layers)
        batch_size = x.shape[0]
        conv1_ = self.conv1(x)
        bn1_ = self.bn1(conv1_)
        relu_ = self.relu(bn1_)
        maxpool_ = self.maxpool(relu_)

        layer1_ = self.layer1(maxpool_)
        layer2_ = self.layer2(layer1_)
        layer3_ = self.layer3(layer2_)
        layer4_ = self.layer4(layer3_)

        avgpool_ = self.avgpool(layer4_)
        flat_ = torch.flatten(avgpool_, 1)
        fc_ = self.fc(flat_)

        #layers_to_use = [conv1_, layer1_, layer2_, layer3_, layer4_, fc_]
        layers_to_use = [conv1_, relu_, layer1_, layer2_, layer3_, layer4_]


        if self._return_layers:
            for layer in layers_to_use:
                print('layer shape', layer.shape)
            #out = torch.cat(list(map(lambda y: y.reshape(batch_size, -1) / torch.sqrt(y.numel()/batch_size), layers_to_use)), 1)
            out = list(map(lambda y: y.reshape(batch_size, -1) / torch.tensor(y.numel() / batch_size).sqrt(), layers_to_use))
            print('out shape', len(out))
            return out
        else:
            return fc_

class PretrainedModel():
    def __init__(self, modelname, pretrained=True):
        #super(PretrainedModel, self).__init__()
        model_pt = model_class_dict[modelname](pretrained=pretrained)
        #model.eval()
        self.model = nn.DataParallel(model_pt.cuda())
        self.model.eval()
        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).cuda()
    def predict(self, x):
        out = (x - self.mu) / self.sigma
        return self.model(out)
    def forward(self, x):
        out = (x - self.mu) / self.sigma
        return self.model(out)
    def __call__(self, x):
        return self.predict(x)
class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()
        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))
    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std
def normalize_model(model: nn.Module, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> nn.Module:
    layers = OrderedDict([
        ('normalize', ImageNormalizer(mean, std)),
        ('model', model)
    ])
    return nn.Sequential(layers)
def Engstrom2019RobustnessNet(ckpt):
    '''def __init__(self):
        #super(Engstrom2019RobustnessNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=1000)
        model_pt = model_class_dict['pt_resnet'](pretrained=False)
        #super
        #self.model = nn.DataParallel(model_pt.cuda().eval())
        self.model = model_pt.cuda().eval()
        self.model.eval()
        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).cuda()
    def forward(self, x, return_features=False):
        x = (x - self.mu) / self.sigma
        #return super(Engstrom2019RobustnessNet, self).forward(x, return_features=return_features)
        return self.model(x)
    def __call__(self, x):
        return self.forward(x)'''
    ##model_pt = model_class_dict['pt_resnet'](pretrained=False).cuda()
    print('loading pt_resnet_return_layers')
    model_pt = model_class_dict['pt_resnet_return_layers'](pretrained=False).cuda()
    model_pt.eval()
    model_pt.load_state_dict(ckpt, strict=True)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalized_model = normalize_model(model=model_pt, mean=mean, std=std)
    normalized_model.eval()
    print('model created')
    normalized_model.cuda()
    return normalized_model
models_Linf = OrderedDict([
    ('Engstrom2019Robustness', {
        'model': Engstrom2019RobustnessNet,
        'data': 'imagenet_linf_4.pt',
        'ckpt_var': None,
    }),
])
models_L2 = OrderedDict([
    ('Engstrom2019Robustness', {
        'model': Engstrom2019RobustnessNet,
        'data': 'imagenet_l2_3_0.pt',
        'ckpt_var': None
    }),
])
other_models = OrderedDict()
model_dicts = {'linf': models_Linf, 'l2': models_L2, 'other': other_models}

def get_base_dir():
    machine_name = socket.gethostname()

    base = '/scratch/vboreiko87/projects/ACSM' #'/mnt/SHARED/ACSM'
    return base

def load_model(modelname, modeldir=f'{get_base_dir()}/ImageNet1000Models/Madry', norm='other'):
    modeldir = '/mnt/SHARED/fcroce42/robust-ensembles/models/models_imagenet'
    model_det = model_dicts[norm][modelname]
    #model = model_det['model']()
    ckpt = torch.load('{}/{}'.format(modeldir, model_det['data']), map_location='cpu')
    '''if not model_det['ckpt_var'] is None:
        ckpt = ckpt[model_det['ckpt_var']]
    try:
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        model.cuda()
        print('standard loading')
    except:
        model.model.load_state_dict(ckpt, strict=True)
        model.model.eval()'''
    model = model_det['model'](ckpt)
    print('model loaded')
    print(type(model))
    return model