import time

from torch.utils.data import DataLoader
import torch

from pnpxai.explainers import AVAILABLE_EXPLAINERS
from tutorials.helpers import get_torchvision_model, get_imagenet_dataset


class TestRobustness():
    def test_computation_time(self):
        batch_size = 64
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model, transform = get_torchvision_model("resnet18")
        model = model.to(device)
        data = get_imagenet_dataset(transform, subset_size=100)
        loader = DataLoader(data, batch_size=batch_size)

        for explainer_type in AVAILABLE_EXPLAINERS:
            explainer = explainer_type(model)

            for input, target in loader:
                start_time = time.time()
                input = input.to(device)
                target = target.to(device)
                explainer.attribute(input, target)
                elapsed = (time.time() - start_time) / batch_size
                assert elapsed < 1
