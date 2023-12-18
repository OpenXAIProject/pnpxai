import torch
from torch.utils.data import DataLoader
from pnpxai import Project
from pnpxai.evaluator import XaiEvaluator
from pnpxai.evaluator.mu_fidelity import MuFidelity
from pnpxai.evaluator.sensitivity import Sensitivity
from pnpxai.evaluator.complexity import Complexity
from pnpxai.utils import set_seed
from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image

set_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, transform = get_torchvision_model("resnet18")
model = model.to(device)
dataset = get_imagenet_dataset(transform=transform, subset_size=16)
loader = DataLoader(dataset, batch_size=8)

def input_extractor(x): return x[0].to(device)
def target_extractor(x): return x[1].to(device)

# vis test
proj = Project("vis test")
expr = proj.create_auto_experiment(
    model=model,
    data=loader,
    input_extractor=input_extractor,
    target_extractor=target_extractor,
    input_visualizer=denormalize_image,
)

# force evaluator
expr.evaluator = XaiEvaluator(metrics=[
    MuFidelity(n_perturbations=1),
    Sensitivity(n_iter=1),
    Complexity(),
])

expr.run()
vis = expr.visualize()