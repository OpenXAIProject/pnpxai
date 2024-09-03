import torch
from torch.utils.data import DataLoader

from pnpxai.utils import set_seed
from pnpxai.explainers import LRPEpsilonPlus
from pnpxai.core.modality import ImageModality
from pnpxai.explainers.utils.postprocess import Identity
from pnpxai.evaluator.metrics import MuFidelity, Sensitivity, Complexity, AbPC
from pnpxai import Experiment

from helpers import get_imagenet_dataset, get_torchvision_model

set_seed(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, transform = get_torchvision_model("resnet18")
model = model.to(device)
explainer = LRPEpsilonPlus(model=model, epsilon=1e-6, n_classes=1000)

dataset = get_imagenet_dataset(transform=transform, subset_size=8)
loader = DataLoader(dataset, batch_size=8)
inputs, targets = next(iter(loader))
inputs, targets = inputs.to(device), targets.to(device)

attrs = explainer.attribute(inputs, targets)

metrics = [
    MuFidelity(model, explainer, n_perturb=10),
    Sensitivity(model, explainer, n_iter=10),
    Complexity(model, explainer)
]

experiment = Experiment(
    model=model,
    data=loader,
    modality=ImageModality(),
    explainers=[explainer],
    postprocessors=[Identity()],
    metrics=metrics,
    input_extractor=lambda x: x[0].to(device),
    label_extractor=lambda x: x[-1].to(device),
    target_extractor=lambda outputs: outputs.argmax(-1).to(device)
)

data_ids = range(4)
for metric_id, metric in enumerate(metrics):
    data = experiment.run_batch(
        data_ids,
        metric_id=metric_id,
        explainer_id=0,
        postprocessor_id=0
    )
    evaluations = data['evaluation']
    for metric, metric_evals in zip(metrics, evaluations):
        print(f"Metric {metric} evaluations:")
        print(metric_evals)
        print()
