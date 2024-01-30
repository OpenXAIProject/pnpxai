import time

from torch.utils.data import DataLoader
import torch
import torchvision

from pnpxai.explainers import AVAILABLE_EXPLAINERS
from pnpxai.tests.helpers import get_dummy_imagenet_dataset


class TestRobustness():
    def test_computation_time(self):
        batch_size = 64
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = torchvision.models.get_model("resnet18").eval()
        model = model.to(device)
        data = get_dummy_imagenet_dataset(n_samples=100)
        loader = DataLoader(data, batch_size=batch_size)

        for explainer_type in AVAILABLE_EXPLAINERS:
            # TODO: memory issues of RAP on gpu
            if torch.cuda.is_available() and explainer_type.__name__ == "RAP":
                continue
            explainer = explainer_type(model)

            for input, target in loader:
                start_time = time.time()
                input = input.to(device)
                target = target.to(device)
                explainer.attribute(input, target)
                elapsed = (time.time() - start_time) / batch_size
                assert elapsed < 1
