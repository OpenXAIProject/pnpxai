from functools import partial
from typing import Optional, Union, Sequence

import torch

from open_xai.core._types import Model
from open_xai.explainers._explainer import Explainer
from open_xai.explainers import IntegratedGradients, RAP

from ._core import ModelArchitecture


class Detector:
    def __init__(self):
        pass

    def __call__(self, model: Model) -> Explainer:
        return RAP(model)
