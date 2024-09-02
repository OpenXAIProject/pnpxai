from typing import Callable, Optional, Tuple, List

from torch.nn.modules import Module
from torch.utils.data import DataLoader

from pnpxai.core._types import Model, DataSource
from pnpxai.core.experiment.experiment import Experiment
from pnpxai.core.modality.modality import Modality, ImageModality, TextModality, TimeSeriesModality, ImageTextModality
from pnpxai.core.recommender import XaiRecommender
from pnpxai.explainers.types import TargetLayer
from pnpxai.evaluator.metrics import PIXEL_FLIPPING_METRICS
from pnpxai.evaluator.metrics import (
    MuFidelity,
    Sensitivity,
    Complexity,
    MoRF,
    LeRF,
    AbPC,
)


METRICS_BASELINE_FN_REQUIRED = PIXEL_FLIPPING_METRICS
METRICS_CHANNEL_DIM_REQUIRED = PIXEL_FLIPPING_METRICS
DEFAULT_METRICS = [
    AbPC,
    MoRF,
    LeRF,
    MuFidelity,
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
        modality: Modality,
        input_extractor: Optional[Callable] = None,
        label_extractor: Optional[Callable] = None,
        target_extractor: Optional[Callable] = None,
        target_labels: bool = False
    ):
        self.recommended = XaiRecommender().recommend(modality=modality, model=model)
        self.modality = modality

        super().__init__(
            model=model,
            data=data,
            modality=modality,
            explainers=self._load_default_explainers(model),
            postprocessors=self._load_default_postprocessors(),
            metrics=self._load_default_metrics(model),
            input_extractor=input_extractor,
            label_extractor=label_extractor,
            target_extractor=target_extractor,
            target_labels=target_labels,
        )

    def _load_default_explainers(self, model):
        explainers = []
        for explainer_type in self.recommended.explainers:
            explainer = explainer_type(model=model)
            default_kwargs = self._generate_default_kwargs_for_explainer()
            for k, v in default_kwargs.items():
                if hasattr(explainer, k):
                    explainer = explainer.set_kwargs(**{k: v})
            explainers.append(explainer)
        return explainers

    def _load_default_metrics(self, model):
        empty_metrics = []  # empty means that explainer is not assigned yet
        for metric_type in DEFAULT_METRICS:
            metric = metric_type(model=model)
            default_kwargs = self._generate_default_kwargs_for_metric()
            for k, v in default_kwargs.items():
                if hasattr(metric, k):
                    metric = metric.set_kwargs(**{k: v})
            empty_metrics.append(metric)
        return empty_metrics

    def _load_default_postprocessors(self):
        return self.modality.get_default_postprocessors()

    def _generate_default_kwargs_for_explainer(self):
        return {
            'feature_mask_fn': self.modality.get_default_feature_mask_fn(),
            'baseline_fn': self.modality.get_default_baseline_fn(),
        }

    def _generate_default_kwargs_for_metric(self):
        return {
            'baseline_fn': self.modality.get_default_baseline_fn(),
            'channel_dim': self.modality.channel_dim,
        }


class AutoExplanationForImageClassification(AutoExplanation):
    def __init__(
        self,
        model: Module,
        data: DataLoader,
        input_extractor: Optional[Callable] = None,
        label_extractor: Optional[Callable] = None,
        target_extractor: Optional[Callable] = None,
        target_labels: bool = False,
        channel_dim: int = 1,
    ):
        super().__init__(
            model=model,
            data=data,
            modality=ImageModality(channel_dim),
            input_extractor=input_extractor,
            label_extractor=label_extractor,
            target_extractor=target_extractor,
            target_labels=target_labels
        )


class AutoExplanationForTextClassification(AutoExplanation):
    def __init__(
        self,
        model: Module,
        data: DataLoader,
        layer: TargetLayer,
        mask_token_id: int,
        input_extractor: Optional[Callable] = None,
        forward_arg_extractor: Optional[Callable] = None,
        additional_forward_arg_extractor: Optional[Callable] = None,
        label_extractor: Optional[Callable] = None,
        target_extractor: Optional[Callable] = None,
        target_labels: bool = False,
        channel_dim: int = -1,
    ):
        self.layer = layer
        self.mask_token_id = mask_token_id
        self.forward_arg_extractor = forward_arg_extractor
        self.additional_forward_arg_extractor = additional_forward_arg_extractor

        super().__init__(
            model=model,
            data=data,
            modality=TextModality(channel_dim),
            input_extractor=input_extractor,
            label_extractor=label_extractor,
            target_extractor=target_extractor,
            target_labels=target_labels
        )

    def _generate_default_kwargs_for_explainer(self):
        return {
            'layer': self.layer,
            'forward_arg_extractor': self.forward_arg_extractor,
            'additional_forward_arg_extractor': self.additional_forward_arg_extractor,
            'feature_mask_fn': self.modality.get_default_feature_mask_fn(),
            'baseline_fn': self.modality.get_default_baseline_fn(self.mask_token_id),
        }

    def _generate_default_kwargs_for_metric(self):
        return {
            'baseline_fn': self.modality.get_default_baseline_fn(self.mask_token_id),
            'channel_dim': self.modality.channel_dim,
        }


class AutoExplanationForVisualQuestionAnswering(AutoExplanation):
    def __init__(
        self,
        model: Module,
        data: DataLoader,
        layer: List[TargetLayer],
        mask_token_id: int,
        input_extractor: Optional[Callable] = None,
        forward_arg_extractor: Optional[Callable] = None,
        additional_forward_arg_extractor: Optional[Callable] = None,
        label_extractor: Optional[Callable] = None,
        target_extractor: Optional[Callable] = None,
        target_labels: bool = False,
        channel_dim: Tuple[int] = (1, -1),
    ):
        self.layer = layer
        self.mask_token_id = mask_token_id
        self.forward_arg_extractor = forward_arg_extractor
        self.additional_forward_arg_extractor = additional_forward_arg_extractor

        super().__init__(
            model=model,
            data=data,
            modality=ImageTextModality(channel_dim),
            input_extractor=input_extractor,
            label_extractor=label_extractor,
            target_extractor=target_extractor,
            target_labels=target_labels
        )

    def _generate_default_kwargs_for_explainer(self):
        return {
            'layer': self.layer,
            'forward_arg_extractor': self.forward_arg_extractor,
            'additional_forward_arg_extractor': self.additional_forward_arg_extractor,
            'feature_mask_fn': self.modality.get_default_feature_mask_fn(),
            'baseline_fn': self.modality.get_default_baseline_fn(self.mask_token_id)
        }

    def _generate_default_kwargs_for_metric(self):
        return {
            'baseline_fn': self.modality.get_default_baseline_fn(self.mask_token_id),
            'channel_dim': self.modality.channel_dim,
        }


class AutoExplanationForTabularClassification(AutoExplanation):
    def __init__(self):
        pass

    def _check_background_data(self, kwargs):
        if self.modality == 'tabular':
            assert kwargs.get(
                'background_data') is not None, "Must have 'background_data' for tabular modality."


class AutoExplanationForTSClassification(AutoExplanation):
    def __init__(
        self,
        model: Module,
        data: DataLoader,
        input_extractor: Optional[Callable] = None,
        label_extractor: Optional[Callable] = None,
        target_extractor: Optional[Callable] = None,
        target_labels: bool = False,
        sequence_dim: int = -1,
        mask_agg_dim: int = -2,
    ):
        self.mask_agg_dim = mask_agg_dim
        super().__init__(
            model=model,
            data=data,
            modality=TimeSeriesModality(sequence_dim),
            input_extractor=input_extractor,
            label_extractor=label_extractor,
            target_extractor=target_extractor,
            target_labels=target_labels
        )

    def _generate_default_kwargs_for_metric(self):
        return {
            'baseline_fn': self.modality.get_default_baseline_fn(),
            'feature_mask_fn': self.modality.get_default_feature_mask_fn(),
            'channel_dim': self.modality.channel_dim,
            'mask_agg_dim': self.mask_agg_dim,
        }