from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence, Optional

import plotly.graph_objects as go

from open_xai.core._types import Model, DataSource, Args


class Explainer:
    def __init__(self, model: Model):
        self.model = model
        pass

    @abstractmethod
    def attribute(self, inputs: DataSource, target: DataSource, *args: Any, **kwargs: Any) -> DataSource:
        pass

    def format_outputs_for_visualization(self, inputs: DataSource, labels: DataSource, *args, **kwargs) -> Sequence[go.Figure]:
        pass


@dataclass
class ExplainerWArgs:
    explainer: Explainer
    args: Optional[Args] = None

    def has_args(self):
        return self.args is not None