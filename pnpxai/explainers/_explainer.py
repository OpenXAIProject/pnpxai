from abc import abstractmethod
from typing import Any, Optional, Dict, Callable

from pnpxai.core._types import Model, DataSource, Task, Tensor
from pnpxai.explainers.utils.post_process import postprocess_attr


class Explainer:
    def __init__(
        self,
        model: Model,
    ):
        self.model = model

    @property
    def device(self):
        return next(self.model.parameters()).device

    @abstractmethod
    def attribute(self, inputs: DataSource, targets: DataSource, **kwargs) -> DataSource:
        pass

    def format_outputs_for_visualization(
        self,
        inputs: Tensor,
        targets: Tensor,
        explanations: Tensor,
        task: Task,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        return postprocess_attr(
            attr=explanations,
            # sign="absolute"
        )


class ExplainerWArgs():
    def __init__(self, explainer: Explainer, kwargs: Optional[Dict[str, Any]] = None):
        self.explainer = explainer
        self.kwargs = kwargs or {}

    def attribute(self, inputs: DataSource, targets: DataSource, **kwargs) -> DataSource:
        kwargs = {
            **self.kwargs,
            "inputs": inputs,
            "targets": targets,
            **kwargs,
        }
        attributions = self.explainer.attribute(**kwargs)
        return attributions

    def format_outputs_for_visualization(
        self,
        inputs: Tensor,
        targets: Tensor,
        explanations: Tensor,
        task: Task,
        kwargs: Optional[dict] = None,
    ):
        kwargs = {
            **self.kwargs,
            **(kwargs or {})
        }
        return self.explainer.format_outputs_for_visualization(
            inputs=inputs,
            targets=targets,
            explanations=explanations,
            task=task,
            kwargs=kwargs,
        )
