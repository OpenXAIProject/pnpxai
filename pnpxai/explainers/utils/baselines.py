from typing import Literal

import torch
import torchvision.transforms.functional as TF


DEFAULT_BASELINE_METHODS = {
    'zeros': torch.zeros_like,
}

