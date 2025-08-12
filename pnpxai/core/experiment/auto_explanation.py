from typing import Callable, Optional, List, Union, Any

import torch

from pnpxai.core._types import Model, DataSource
from pnpxai.core.experiment.experiment import Experiment
from pnpxai.core.modality.modality import Modality
from pnpxai.core.recommender import XaiRecommender
from pnpxai.explainers.types import TargetLayerOrTupleOfTargetLayers
from pnpxai.evaluator.metrics import (
    MoRF,
    LeRF,
    AbPC,
)
from pnpxai.utils import _camel_to_snake


DEFAULT_METRICS = [
    MoRF,
    LeRF,
    AbPC,
]


class AutoExplanation(Experiment):
    """
    An extension of Experiment class with automatic explainers and parameters recommendation.

    Parameters:
        model (Model): The machine learning model to be analyzed.
        data (DataSource): The data source used for the experiment.
        modality (Modality): An object to specify modality-specific workflow.
        input_extractor (Optional[Callable], optional): Custom function to extract input features.
        label_extractor (Optional[Callable], optional): Custom function to extract labels features.
        target_extractor (Optional[Callable], optional): Custom function to extract target features.
        input_visualizer (Optional[Callable], optional): Custom function for visualizing input features.
        target_labels (Optional[bool]): Whether to use target labels.

    Attributes:
        recommended (RecommenderOutput): A data object, containing recommended explainers.
    """

    def __init__(
        self,
        model: Model,
        data: DataSource,
        modality: Modality,
        target_layer: Optional[TargetLayerOrTupleOfTargetLayers] = None,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], torch.Tensor]] = None,
        target_class_extractor: Optional[Callable[[Any], Any]] = None,
        label_key: Optional[Union[str, int]] = -1,
        target_labels: bool = False,
        cache_device: Optional[Union[torch.device, str]] = None,
    ):
        super().__init__(
            model=model,
            data=data,
            modality=modality,
            target_layer=target_layer,
            target_input_keys=target_input_keys,
            additional_input_keys=additional_input_keys,
            output_modifier=output_modifier,
            target_class_extractor=target_class_extractor,
            label_key=label_key,
            target_labels=target_labels,
            cache_device=cache_device,
        )
        self.recommended = XaiRecommender().recommend(
            modality=modality,
            model=model,
        )
        self._load_default_explainers()
        self._load_default_metrics()

    def _load_default_explainers(self):
        for explainer_type in self.recommended.explainers:
            self.explainers.add(key=explainer_type.alias[0], value=explainer_type)

    def _load_default_metrics(self):
        for tp in DEFAULT_METRICS:
            self.metrics.add(key=tp.alias[0], value=tp)
