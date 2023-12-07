from abc import abstractmethod
from typing import Any, Optional, Dict, Callable

from pnpxai.core._types import Model, DataSource, Task
from pnpxai.explainers.utils.post_process import postprocess_attr


class Explainer:
    def __init__(
        self,
        model: Model,
    ):
        self.model = model
        self.device = next(self.model.parameters()).device

    @abstractmethod
    def attribute(self, inputs: DataSource, targets: DataSource, **kwargs) -> DataSource:
        pass

    def format_outputs_for_visualization(
        self,
        data: DataSource,
        explanations: DataSource,
        task: Task,
        input_extractor: Callable[[Any], Any],
        target_extractor: Callable[[Any], Any],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        return postprocess_attr(
            attr=explanations,
            sign="absolute"
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
        print(self.explainer)
        attributions = self.explainer.attribute(**kwargs)
        return attributions

    def format_outputs_for_visualization(
        self,
        data: DataSource,
        explanations: DataSource,
        input_extractor: Callable[[Any], Any],
        target_extractor: Callable[[Any], Any],
        task: Task,
        kwargs,
    ):
        kwargs = {
            **self.kwargs,
            **kwargs
        }
        return self.explainer.format_outputs_for_visualization(
            data=data,
            input_extractor=input_extractor,
            target_extractor=target_extractor,
            explanations=explanations,
            task=task,
            kwargs=kwargs,
        )
