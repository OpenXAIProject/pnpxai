from time import time_ns

from torch.utils.data import DataLoader
from pnpxai.explainers import GuidedGradCam, ExplainerWArgs
from pnpxai.evaluator import XaiEvaluator
from pnpxai.evaluator.mu_fidelity import MuFidelity
from pnpxai.evaluator.sensitivity import Sensitivity
from pnpxai.evaluator.complexity import Complexity

from helpers import get_imagenet_dataset, get_torchvision_model

model, transform = get_torchvision_model("inception_v3")
dataset = get_imagenet_dataset(transform=transform, subset_size=8)
loader = DataLoader(dataset, batch_size=1)
inputs, labels = next(iter(loader))
explainer = ExplainerWArgs(
    explainer = GuidedGradCam(model)
)
attrs = explainer.attribute(inputs, labels)

evaluator = XaiEvaluator(metrics=[
    MuFidelity(n_perturbations=10, batch_size=4),
    Sensitivity(n_iter=1),
    Complexity(n_bins=10)
])
evals = evaluator(inputs=inputs, targets=labels, explainer_w_args=explainer, explanations=attrs)