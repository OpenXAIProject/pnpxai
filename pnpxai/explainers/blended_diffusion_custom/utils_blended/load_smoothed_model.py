import torch

from models.wideresnet import WideResNet
from models.shake_shake import ShakeNet
from models.cifar_resnet import ResNet

from torch.nn import Sequential, Module


class NormalizeInput(Module):
    def __init__(self, mean=(0.4914, 0.4822, 0.4465),
                 std=(0.2023, 0.1994, 0.2010)):
        super().__init__()

        self.register_buffer('mean', torch.Tensor(mean).reshape(1, -1, 1, 1))
        self.register_buffer('std', torch.Tensor(std).reshape(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def get_model(name, num_classes=10, normalize_input=False):
    name_parts = name.split('-')
    if name_parts[0] == 'wrn':
        depth = int(name_parts[1])
        widen = int(name_parts[2])
        model = WideResNet(
            depth=depth, num_classes=num_classes, widen_factor=widen)

    elif name_parts[0] == 'ss':
        model = ShakeNet(dict(depth=int(name_parts[1]),
                              base_channels=int(name_parts[2]),
                              shake_forward=True, shake_backward=True,
                              shake_image=True, input_shape=(1, 3, 32, 32),
                              n_classes=num_classes,
                              ))
    elif name_parts[0] == 'resnet':
        model = ResNet(num_classes=num_classes, depth=int(name_parts[1]))
    else:
        raise ValueError('Could not parse model name %s' % name)

    if normalize_input:
        model = Sequential(NormalizeInput(), model)

    return model

def load_smoothed_model(path, model):
    # load the base classifier
    checkpoint = torch.load(path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    if not all(k.startswith('module') for k in state_dict):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    num_classes = checkpoint.get('num_classes', 10)
    normalize_input = checkpoint.get('normalize_input', False)

    base_classifier = get_model(model, num_classes=num_classes,
                            normalize_input=normalize_input)
    base_classifier = torch.nn.DataParallel(base_classifier).cuda()
    # setting loader to be non-strict so we can load Cohen et al.'s model
    base_classifier.load_state_dict(state_dict,
                                strict=(model != 'resnet-110'))

    # create the smooothed classifier g
    #smoothed_classifier = Smooth(base_classifier, num_classes, sigma)

    return base_classifier