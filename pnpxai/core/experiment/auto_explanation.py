from typing import Literal, Callable, Optional, Type, Union, Tuple

from pnpxai.core.experiment.experiment import Experiment
from pnpxai.core.detector import detect_model_architecture
from pnpxai.core.recommender import XaiRecommender
from pnpxai.core._types import DataSource, Model, ModalityOrTupleOfModalities, Modality
from pnpxai.explainers.types import TargetLayerOrListOfTargetLayers, ForwardArgumentExtractor
from pnpxai.explainers.utils.baselines import get_default_baseline_function
from pnpxai.explainers.utils.feature_masks import get_default_feature_mask_fn
from pnpxai.explainers.utils.postprocess import (
    PostProcessor,
    all_postprocessors,
    get_default_channel_dim,
)
from pnpxai.evaluator.metrics import PIXEL_FLIPPING_METRICS
from pnpxai.evaluator.metrics import(
    MuFidelity,
    Sensitivity,
    Complexity,
    MoRF,
    LeRF,
    AbPC,
)
from pnpxai.utils import format_into_tuple


METRICS_BASELINE_FN_REQUIRED = PIXEL_FLIPPING_METRICS
METRICS_CHANNEL_DIM_REQUIRED = PIXEL_FLIPPING_METRICS
DEFAULT_METRICS = [
    MuFidelity,
    AbPC,
    Sensitivity,
    Complexity,
]

class AutoExplanation(Experiment):
    """
    An extension of Experiment class with automatic explainers and evaluation metrics recommendation.

    Parameters:
        model (Model): The machine learning model to be analyzed.
        data (DataSource): The data source used for the experiment.
        task (Literal["image", "tabular"], optional): The task type, either "image" or "tabular". Defaults to "image".
        question (Literal["why", "how"], optional): The type of question the experiment aims to answer, either "why" or "how". Defaults to "why".
        evaluator_enabled (bool, optional): Whether to enable the evaluator. Defaults to True.
        input_extractor (Optional[Callable], optional): Custom function to extract input features. Defaults to None.
        label_extractor (Optional[Callable], optional): Custom function to extract target labels. Defaults to None.
        input_visualizer (Optional[Callable], optional): Custom function for visualizing input features. Defaults to None.
        target_visualizer (Optional[Callable], optional): Custom function for visualizing target labels. Defaults to None.
    """
    def __init__(
        self,
        model: Model,
        data: DataSource,
        modality: ModalityOrTupleOfModalities = "image",
        input_extractor: Optional[Callable] = None,
        label_extractor: Optional[Callable] = None,
        target_extractor: Optional[Callable] = None,
        target_labels: bool = False,
        **kwargs,
    ):
        self.modality = modality
        self._check_layer(kwargs)
        self._check_background_data(kwargs)
        self._check_mask_token_id(kwargs)
        self.recommended = XaiRecommender().recommend(modality=modality, model=model)
        super().__init__(
            model=model,
            data=data,
            explainers=self._load_default_explainers(model, kwargs),
            postprocessors=self._load_default_postprocessors(kwargs),
            metrics=self._load_default_metrics(model, kwargs),
            input_extractor=input_extractor,
            label_extractor=label_extractor,
            target_extractor=target_extractor,
            target_labels=target_labels,
            modality=modality,
            mask_token_id=kwargs.get('mask_token_id'),
        )

    def _check_layer(self, kwargs):
        modality = format_into_tuple(self.modality)
        if 'text' in modality:
            assert kwargs.get('layer'), "Must have 'layer' for text modality. It might be a word embedding layer of your model."
    
    def _check_background_data(self, kwargs):
        if self.modality == 'tabular':
            assert kwargs.get('background_data') is not None, "Must have 'background_data' for tabular modality."

    def _check_mask_token_id(self, kwargs):
        modality = format_into_tuple(self.modality)
        if 'text' in modality:
            assert kwargs.get('mask_token_id'), "Must have 'mask_token_id' for text modality."

    def _load_default_explainers(self, model, kwargs):
        # explainers
        explainers = []
        for explainer_type in self.recommended.explainers:
            explainer = explainer_type(model=model)
            default_kwargs = self._generate_default_kwargs_for_explainer(kwargs)
            for k, v in default_kwargs.items():
                if hasattr(explainer, k):
                    explainer = explainer.set_kwargs(**{k: v})
            explainers.append(explainer)
        return explainers

    def _load_default_postprocessors(self, kwargs):
        channel_dim = kwargs.get('channel_dim') or get_default_channel_dim(self.modality)
        return all_postprocessors(channel_dim)

    def _load_default_metrics(self, model, kwargs):
        channel_dim = kwargs.get('channel_dim') or get_default_channel_dim(self.modality)
        empty_metrics = []
        for metric_type in DEFAULT_METRICS:
            metric = metric_type(model=model)
            default_kwargs = self._generate_default_kwargs_for_metric(kwargs)
            for k, v in default_kwargs.items():
                if hasattr(metric, k):
                    metric = metric.set_kwargs(**{k: v})
            empty_metrics.append(metric)
        return empty_metrics

    def _generate_default_kwargs_for_explainer(self, kwargs):
        return {
            'layer': kwargs.get('layer'),
            'background_data': kwargs.get('background_data'),
            'forward_arg_extractor': kwargs.get('forward_arg_extractor'),
            'additional_forward_arg_extractor': kwargs.get('additional_forward_arg_extractor'),
            'feature_mask_fn': kwargs.get('feature_mask_fn') \
                or get_default_feature_mask_fn(self.modality),
            'baseline_fn': kwargs.get('baseline_fn') \
                or get_default_baseline_function(self.modality, mask_token_id=kwargs.get('mask_token_id')),
        }

    def _generate_default_kwargs_for_metric(self, kwargs):
        return {
            'baseline_fn': kwargs.get('baseline_fn') \
                or get_default_baseline_function(self.modality, mask_token_id=kwargs.get('mask_token_id')),
            'channel_dim': kwargs.get('channel_dim') \
                or get_default_channel_dim(self.modality)
        }

