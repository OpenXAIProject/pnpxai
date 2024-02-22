import torch
from torch.utils.data import DataLoader

from pnpxai.utils import set_seed
from pnpxai.explainers import LRP, ExplainerWArgs
from pnpxai.evaluator import MuFidelity, Sensitivity, Complexity
from pnpxai import Experiment

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

metrics = [
    MuFidelity(n_perturbations=10), Sensitivity(n_iter=10), Complexity()
]

experiment = Experiment(
    model=model,
    data=loader,
    explainers=[explainer],
    metrics=metrics,
    input_extractor=lambda x: x[0].to(device)
)

experiment.run()
evaluations = experiment.get_evaluations_flattened()[0]
for metric, metric_evals in zip(metrics, evaluations):
    print(f"Metric {metric} evaluations:")
    print(metric_evals)
    print()