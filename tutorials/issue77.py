import torch
from torch.utils.data import DataLoader
from pnpxai import Project
from pnpxai.explainers import LRP, ExplainerWArgs
from pnpxai.evaluator.sensitivity import Sensitivity
from helpers import get_imagenet_dataset, get_torchvision_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, transform = get_torchvision_model("resnet18")
model = model.to(device)
def input_extractor(x): return x[0].to(device)
def target_extractor(x): return x[1].to(device)

dataset = get_imagenet_dataset(transform=transform, subset_size=8)
loader = DataLoader(dataset, batch_size=1)
expr = Project("test_proj").create_experiment(
    model=model,
    data=loader,
    explainers=[ExplainerWArgs(LRP(model))],
    metrics=[Sensitivity()],
    input_extractor=input_extractor,
    target_extractor=target_extractor
)
expr.run()