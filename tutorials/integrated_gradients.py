from torch.utils.data import DataLoader

from pnpxai.explainers import IntegratedGradients, ExplainerWArgs
from pnpxai.evaluator import XaiEvaluator
from pnpxai.evaluator.mu_fidelity import MuFidelity
from pnpxai.evaluator.sensitivity import Sensitivity
from pnpxai.evaluator.complexity import Complexity

from helpers import get_imagenet_dataset, get_torchvision_model

model, transform = get_torchvision_model("resnet18")
dataset = get_imagenet_dataset(transform=transform, subset_size=100)
loader = DataLoader(dataset, batch_size=8)
inputs, labels = next(iter(loader))
targets = labels

# explanation
explainer = ExplainerWArgs(
    explainer=IntegratedGradients(model)
)
attrs = explainer.attribute(inputs, targets, n_classes=1000)

# evaluation
metrics = [MuFidelity(n_perturbations=10), Sensitivity(n_iter=1), Complexity(n_bins=10)]
evaluator = XaiEvaluator(metrics=metrics)
evals = evaluator(
    inputs=inputs,
    targets=targets,
    explainer_w_args=explainer,
    explanations=attrs
)