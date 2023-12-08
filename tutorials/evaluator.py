import torch
from torch.utils.data import DataLoader

from pnpxai.utils import set_seed
from pnpxai.explainers import LRP, ExplainerWArgs
from pnpxai.evaluator import XaiEvaluator
from pnpxai.evaluator.mu_fidelity import MuFidelity
from pnpxai.evaluator.sensitivity import Sensitivity

from helpers import get_imagenet_dataset, get_torchvision_model

set_seed(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, transform = get_torchvision_model("resnet18")
model = model.to(device)
explainer = ExplainerWArgs(
    explainer=LRP(model=model),
    kwargs={"epsilon": 1e-6, "n_classes": 1000},
)

dataset = get_imagenet_dataset(transform=transform, subset_size=8)
loader = DataLoader(dataset, batch_size=8)
inputs, targets = next(iter(loader))
inputs, targets = inputs.to(device), targets.to(device)

attrs = explainer.attribute(inputs, targets)

# import pdb; pdb.set_trace()
mufd = MuFidelity()
mufd(model, explainer, inputs, targets, attrs)
