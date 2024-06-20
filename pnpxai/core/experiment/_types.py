from dataclasses import dataclass, asdict
from functools import partial
from typing import Dict, Callable, Type, Optional, Union, Tuple

from torch import nn
from torch.utils.data import Dataset

from pnpxai.core._types import Model, Task, Modality
from pnpxai.explainers.base import Explainer
from pnpxai.metrics.base import Metric


@dataclass
class ModelInfo:
    model: Model
    task: Task


@dataclass
class DatasetInfo:
    modality: Modality
    dataset: Dataset
    input_extractors: Optional[Tuple[Callable]]=(lambda data: data[0],)
    label_extractor: Optional[Tuple[Callable]]=lambda data: data[-1],

    def to_dict(self):
        return asdict(self)


@dataclass
class ExplainerInfo:
    explainer_type: Type[Explainer]
    explainer_kwargs: Optional[Dict]=None
    attribute_kwargs: Optional[Dict]=None

    def to_explainer(self, model):
        kwargs = {} if self.explainer_kwargs is None else self.explainer_kwargs
        return self.explainer_type(model, **kwargs)
    
    def to_explain_func(self, model):
        kwargs = {} if self.attribute_kwargs is None else self.attribute_kwargs
        return partial(
            self.to_explainer(model).attribute,
            **kwargs
        )


@dataclass
class MetricInfo:
    metric_type: Type[Metric]
    metric_kwargs: Optional[Dict]=None
    evaluate_kwargs: Optional[Dict]=None

    def to_metric(self, model):
        kwargs = {} if self.metric_kwargs is None else self.metric_kwargs
        return self.metric_type(model, **kwargs)
    
    def to_metric_func(self, model):
        kwargs = {} if self.evaluate_kwargs is None else self.evaluate_kwargs
        return partial(
            self.to_metric(model).evaluate,
            **kwargs
        )


@dataclass
class ExperimentInfo:
    model: ModelInfo
    dataset: DatasetInfo
    explainers: Dict[str, ExplainerInfo]
    metrics: Dict[str, MetricInfo]

    @classmethod
    def from_dict(cls, config: Dict):
        return cls(
            model=ModelInfo(**config["model"]),
            dataset=DatasetInfo(**config["dataset"]),
            explainers={
                explainer_nm: ExplainerInfo(**explainer_data)
                for explainer_nm, explainer_data in config["explainers"].items()
            },
            metrics={
                metric_nm: MetricInfo(**metric_data)
                for metric_nm, metric_data in config["metrics"].items()
            }
        )

ModelConfig = Union[ModelInfo, Dict]
DatasetConfig = Union[DatasetInfo, Dict]
ExplainerConfig = Union[ExplainerInfo, Dict]
MetricConfig = Union[MetricInfo, Dict]
ExperimentConfig = Union[ExperimentInfo, Dict]