from abc import abstractmethod
from typing import Any
from xai_pnp.core._types import Model, DataSource


class Explainer:
    def __init__(self):
        pass

    @abstractmethod
    def run(self, model: Model, data: DataSource, *args: Any, **kwargs: Any) -> Any:
        pass

    def format_outputs_for_visualization(self, outputs: Any):
        pass
