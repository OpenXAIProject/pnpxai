import time

from torch.utils.data import DataLoader
import torch
import torchvision

from pnpxai.explainers import ExplainerWArgs
from pnpxai.explainers.lrp import LRP
from pnpxai.evaluator.complexity import Complexity
from pnpxai.evaluator.mu_fidelity import MuFidelity
from pnpxai.evaluator.sensitivity import Sensitivity
from pnpxai.tests.helpers import get_dummy_imagenet_dataset

METRICS_LIMIT_MAP = {
    Complexity: 1,
    MuFidelity: 1,
    Sensitivity: 1,
}


class TestRobustness():
    def test_computation_time(self):
        batch_size = 64
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torchvision.models.get_model("resnet18").eval()
        model = model.to(device)
        explainer_w_args = ExplainerWArgs(LRP(model))
        data = get_dummy_imagenet_dataset(n_samples=100)
        loader = DataLoader(data, batch_size=batch_size)

        for metric_type in [Complexity, MuFidelity, Sensitivity]:
            metric = metric_type()

            for input, target in loader:
                start_time = time.time()
                input = input.to(device)
                target = target.to(device)
                metric(model, explainer_w_args, input)
                elapsed = (time.time() - start_time) / batch_size
                assert elapsed < METRICS_LIMIT_MAP[metric_type]
