from abc import abstractmethod
from typing import Any, Sequence

import plotly.graph_objects as go
from torch import nn
from open_xai.core._types import Model, DataSource


class Explainer:
    def __init__(self, model: Model):
        self.model = model
        pass

    @abstractmethod
    def run(self, data: DataSource, *args: Any, **kwargs: Any) -> DataSource:
        pass

    def format_outputs_for_visualization(self, inputs: DataSource, outputs: DataSource, *args, **kwargs) -> Sequence[go.Figure]:
        pass
