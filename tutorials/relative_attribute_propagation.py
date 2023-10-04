from functools import partial

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.models.vgg import vgg11_bn
import torchvision.transforms as transforms

from xai_pnp import Project
from xai_pnp.explainers import RAP


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_loader = DataLoader(
    datasets.ImageFolder(
        './data/imagenet/',
        transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ])),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = vgg11_bn(pretrained=True).to(device)
explainer = RAP(model)

def vgg_relprop(self, r: torch.Tensor):
    x1 = self.model.classifier.rule.relprop(r)
    if torch.is_tensor(x1) == False:
        for i in range(len(x1)):
            x1[i] = x1[i].reshape_as(next(reversed(self.model.features._modules.values())).rule.Y)
    else:
        x1 = x1.reshape_as(next(reversed(self.model.features._modules.values())).rule.Y)
    x1 = self.model.avgpool.rule.relprop(x1)
    x1 = self.model.features.rule.relprop(x1)

    return x1

explainer.method.relprop = partial(vgg_relprop, explainer.method)

project = Project('test_project')
exp = project.explain(explainer)
exp.run(val_loader)

print(project.experiments[0].runs[0].outputs)