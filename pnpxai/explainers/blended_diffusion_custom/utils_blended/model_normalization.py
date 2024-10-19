import torch
from torchvision import transforms

class NormalizationWrapper(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()

        mean = mean[..., None, None]
        std = std[..., None, None]

        self.train(model.training)

        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x_normalized = (x - self.mean)/self.std
        return self.model(x_normalized)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict()

def IdentityWrapper(model):
    mean = torch.tensor([0., 0., 0.])
    std = torch.tensor([1., 1., 1.])
    return NormalizationWrapper(model, mean, std)

def Cifar10Wrapper(model):
    mean = torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618])
    std = torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628])
    return NormalizationWrapper(model, mean, std)

def Cifar100Wrapper(model):
    mean = torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618])
    std = torch.tensor([0.24703225141799082, 0.24348516474564, 0.26158783926049628])
    return NormalizationWrapper(model, mean, std)

def SVHNWrapper(model):
    mean = torch.tensor([0.4377, 0.4438, 0.4728])
    std = torch.tensor([0.1201, 0.1231, 0.1052])
    return NormalizationWrapper(model, mean, std)

def CelebAWrapper(model):
    mean = torch.tensor([0.5063, 0.4258, 0.3832])
    std = torch.tensor([0.2632, 0.2424, 0.2385])
    return NormalizationWrapper(model, mean, std)

def TinyImageNetWrapper(model):
    mean = torch.tensor([0.4802, 0.4481, 0.3975])
    std = torch.tensor([0.2302, 0.2265, 0.2262])
    return NormalizationWrapper(model, mean, std)

def ImageNetWrapper(model):
    print('Using ImageNetWrapper')
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return NormalizationWrapper(model, mean, std)

def RestrictedImageNetWrapper(model):
    mean = torch.tensor([0.4717, 0.4499, 0.3837])
    std = torch.tensor([0.2600, 0.2516, 0.2575])
    return NormalizationWrapper(model, mean, std)

def BigTransferWrapper(model):
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    return NormalizationWrapper(model, mean, std)

def LSUNScenesWrapper(model):
    #imagenet
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return NormalizationWrapper(model, mean, std)

def FundusKaggleWrapper(model):
    # using distance_check.py (for not background-subtracted!)
    mean = torch.tensor([0.4509289264678955, 0.29734623432159424, 0.20647032558918])
    std = torch.tensor([0.27998650074005127, 0.191138356924057, 0.1482602059841156])
    return NormalizationWrapper(model, mean, std)


class ResizeWrapper(torch.nn.Module):
    def __init__(self, model, size, interpolation=3):
        super().__init__()

        self.train(model.training)

        self.interpolation = interpolation
        self.model = model
        self.size = size

    @property
    def return_layers(self):
        """I'm the 'x' property."""
        print("getter of x called")
        return self.model.model.model._return_layers

    @return_layers.setter
    def return_layers(self, value):
        print("setter of x called")
        self.model.model.model._return_layers = value

    @return_layers.deleter
    def return_layers(self):
        print("deleter of x called")
        del self.model.model.model._return_layers

    def forward(self, x):
        # print('using avgpool')
        # return self.model(torch.nn.AdaptiveAvgPool2d((self.size, self.size))(x))
        return self.model(transforms.Resize(self.size, interpolation=self.interpolation)(x))

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict()


class ResizeAndMeanWrapper(torch.nn.Module):
    def __init__(self, model, size=224, interpolation=3,
                 mean=torch.tensor([0.0, 0.0, 0.0]),
                 std=torch.tensor([1.0, 1.0, 1.0])):
        super().__init__()

        self.train(model.training)

        self.interpolation = interpolation
        self.model = model
        self.size = size

        mean = mean[..., None, None]
        std = std[..., None, None]

        self.train(model.training)

        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @property
    def return_layers(self):
        """I'm the 'x' property."""
        print("getter of x called")
        return self.model.model.model._return_layers

    @return_layers.setter
    def return_layers(self, value):
        print("setter of x called")
        self.model.model.model._return_layers = value

    @return_layers.deleter
    def return_layers(self):
        print("deleter of x called")
        del self.model.model.model._return_layers

    def forward(self, x):
        # print('using avgpool')
        # return self.model(torch.nn.AdaptiveAvgPool2d((self.size, self.size))(x))
        return self.model(
            (transforms.Resize(self.size, interpolation=self.interpolation)(x) - self.mean) / self.std
        )

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict()


def FundusKaggleWrapperBackgroundSubtracted(model):
    # using distance_check.py (for background-subtracted!)
    # Note: the first models were run with FundusKaggleWrapper!
    mean = torch.tensor([0.5038442015647888, 0.5026502013206482, 0.5020962357521057])
    std = torch.tensor([0.06049098074436188, 0.07234558463096619, 0.045092713087797165])
    return NormalizationWrapper(model, mean, std)
