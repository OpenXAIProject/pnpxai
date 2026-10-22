import time

from torch.utils.data import DataLoader
import torch
import torchvision

from pnpxai.explainers import (
    AVAILABLE_EXPLAINERS,
    RAP,
    ATTENTION_SPECIFIC_EXPLAINERS,
    Gfgp,
    MadXai,
)
from tests.helpers import get_dummy_imagenet_dataset


class TestRobustness:
    def test_computation_time(self):
        batch_size = 16
        is_gpu_available = torch.cuda.is_available()
        device = torch.device("cuda:0" if is_gpu_available else "cpu")
        model = torchvision.models.get_model("resnet18").eval()
        model = model.to(device)
        data = get_dummy_imagenet_dataset(n_samples=100)
        loader = DataLoader(data, batch_size=batch_size)

        for explainer_type in AVAILABLE_EXPLAINERS:
            # TODO: memory issues of RAP on gpu
            if (
                is_gpu_available
                and issubclass(explainer_type, (RAP, *ATTENTION_SPECIFIC_EXPLAINERS))
                or issubclass(explainer_type, Gfgp)
                # MadXai explains 1-channel brain MRI with its own diffusion model,
                # not an ImageNet classifier; see docs/_cards/mad_xai.md
                or issubclass(explainer_type, MadXai)
            ):
                continue
            explainer = explainer_type(model)

            inputs, targets = next(iter(loader))
            start_time = time.time()
            inputs = inputs.to(device)
            targets = targets.to(device)
            explainer.attribute(inputs, targets)
            elapsed = (time.time() - start_time) / batch_size
            assert elapsed < 1

            torch.cuda.empty_cache()
            del explainer, inputs, targets
