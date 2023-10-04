from abc import abstractmethod
from typing import Any

from torch import nn
from xai_pnp.core._types import Model, DataSource


class Explainer:
    def __init__(self, model: Model):
        self.model = model
        pass

    @abstractmethod
    def run(self, data: DataSource, *args: Any, **kwargs: Any) -> Any:
        pass

    def format_outputs_for_visualization(self, outputs: Any):
        pass
