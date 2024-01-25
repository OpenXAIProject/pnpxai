import time

from torch.utils.data import DataLoader
import torch

from pnpxai.explainers import AVAILABLE_EXPLAINERS
from tutorials.helpers import get_torchvision_model, get_imagenet_dataset


class TestRobustness():
    def test_robustness(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for explainer_type in AVAILABLE_EXPLAINERS:
            model, transform = get_torchvision_model("resnet18")
            model = model.to(device)
            data = get_imagenet_dataset(transform, subset_size=1000)
            loader = DataLoader(data, batch_size=25)
            explainer = explainer_type(model)

            start_time = time.time()
            for input, target in loader:
                input = input.to(device)
                target = target.to(device)
                explainer.attribute(input, target)
                assert True
            elapsed = time.time() - start_time
            assert elapsed < 1000
            print(elapsed)
            assert False
