import time

from torch.utils.data import DataLoader
import torch
import torchvision

from pnpxai.explainers import LRPUniformEpsilon
from pnpxai.evaluator.metrics.complexity import Complexity
from pnpxai.evaluator.metrics.mu_fidelity import MuFidelity
from pnpxai.evaluator.metrics.sensitivity import Sensitivity

from tests.helpers import get_dummy_imagenet_dataset

METRICS_LIMIT_MAP = {
    Complexity: 1,
    MuFidelity: 1,
    Sensitivity: 1,
}


class TestRobustness:
    def test_computation_time(self):
        batch_size = 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torchvision.models.get_model("resnet18").eval()
        model = model.to(device)
        explainer = LRPUniformEpsilon(model)
        data = get_dummy_imagenet_dataset(n_samples=100)
        loader = DataLoader(data, batch_size=batch_size)

        for metric_type in [Complexity, MuFidelity, Sensitivity]:
            metric = metric_type(model, explainer)

            inputs, targets = next(iter(loader))

            start_time = time.time()

            inputs = inputs.to(device)
            targets = targets.to(device)
            dummy_attrs = torch.ones_like(inputs)

            metric.evaluate(inputs, targets, dummy_attrs)
            elapsed = (time.time() - start_time) / batch_size
            assert elapsed < METRICS_LIMIT_MAP[metric_type]
