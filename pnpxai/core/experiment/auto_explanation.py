from typing import Callable, Optional, Tuple, List
import itertools

from torch.nn.modules import Module
from torch.utils.data import DataLoader

from pnpxai.core._types import Model, DataSource
from pnpxai.core.experiment.experiment import Experiment
from pnpxai.core.modality.modality import Modality, ImageModality, TextModality, TimeSeriesModality
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
from pnpxai.utils import format_into_tuple


METRICS_BASELINE_FN_REQUIRED = PIXEL_FLIPPING_METRICS
METRICS_CHANNEL_DIM_REQUIRED = PIXEL_FLIPPING_METRICS
DEFAULT_METRICS_FOR_TEXT = [
    MoRF,
    LeRF,
    AbPC,
]
DEFAULT_METRICS = [
    MuFidelity,
    AbPC,
    Sensitivity,
    Complexity,
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
        modalities = format_into_tuple(self.modality)
        if len(modalities) == 1:
            return self.modality.get_default_postprocessors()
        return list(itertools.product(*tuple(
            modality.get_default_postprocessors()
            for modality in modalities
        )))

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
    """
    An extension of AutoExplanation class with modality set to the ImageModality.

    Parameters:
        model (Model): The machine learning model to be analyzed.
        data (DataSource): The data source used for the experiment.
        input_extractor (Optional[Callable]): Custom function to extract input features.
        label_extractor (Optional[Callable]): Custom function to extract labels features.
        target_extractor (Optional[Callable]): Custom function to extract target features.
        target_labels (Optional[bool]): Whether to use target labels.
        channel_dim (int): Channel dimension.

    Attributes:
        modality (ImageModality): An object to specify modality-specific workflow.
        recommended (RecommenderOutput): A data object, containing recommended explainers.
    """
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
    """
    An extension of AutoExplanation class with modality set to the TextModality.

    Parameters:
        model (Model): The machine learning model to be analyzed.
        data (DataSource): The data source used for the experiment.
        layer (TargetLayer): A Module or its string representation to select a target layer for analysis.
        mask_token_id (int): A mask token id.
        input_extractor (Optional[Callable], optional): Custom function to extract input features.
        forward_arg_extractor (Optional[Callable]): Custom function to extract forward arguments.
        additional_forward_arg_extractor (Optional[Callable]): Custom function to extract additional forward arguments.
        label_extractor (Optional[Callable], optional): Custom function to extract labels features.
        target_extractor (Optional[Callable], optional): Custom function to extract target features.
        target_labels (Optional[bool]): Whether to use target labels.
        channel_dim (int): Channel dimension.

    Attributes:
        modality (ImageModality): An object to specify modality-specific workflow.
        recommended (RecommenderOutput): A data object, containing recommended explainers.
    """
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
            modality=TextModality(
                channel_dim=channel_dim, mask_token_id=mask_token_id),
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
            'baseline_fn': self.modality.get_default_baseline_fn(),
        }

    def _generate_default_kwargs_for_metric(self):
        return {
            'baseline_fn': self.modality.get_default_baseline_fn(),
            'channel_dim': self.modality.channel_dim,
        }

    def _load_default_metrics(self, model):
        empty_metrics = []  # empty means that explainer is not assigned yet
        for metric_type in DEFAULT_METRICS_FOR_TEXT:
            metric = metric_type(model=model)
            default_kwargs = self._generate_default_kwargs_for_metric()
            for k, v in default_kwargs.items():
                if hasattr(metric, k):
                    metric = metric.set_kwargs(**{k: v})
            empty_metrics.append(metric)
        return empty_metrics


class AutoExplanationForVisualQuestionAnswering(AutoExplanation):
    """
    An extension of AutoExplanation class with multiple modalities, namely ImageModality and TextModality.

    Parameters:
        model (Model): The machine learning model to be analyzed.
        data (DataSource): The data source used for the experiment.
        layer (TargetLayer): A Module or its string representation to select a target layer for analysis.
        mask_token_id (int): A mask token id.
        modality (Modality): An object to specify modality-specific workflow.
        input_extractor (Optional[Callable], optional): Custom function to extract input features.
        forward_arg_extractor (Optional[Callable]): Custom function to extract forward arguments.
        additional_forward_arg_extractor (Optional[Callable]): Custom function to extract additional forward arguments.
        label_extractor (Optional[Callable], optional): Custom function to extract labels features.
        target_extractor (Optional[Callable], optional): Custom function to extract target features.
        target_labels (Optional[bool]): Whether to use target labels.
        channel_dim (Tuple[int]): Channel dimension. Requires a tuple channel dimensions for image and text modalities.

    Attributes:
        modality (Tuple[ImageModality, TextModality]): A tuple of objects to specify modality-specific workflow.
        recommended (RecommenderOutput): A data object, containing recommended explainers.
    """
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
            modality=(
                ImageModality(channel_dim=channel_dim[0]),
                TextModality(channel_dim=channel_dim[1], ),
            ),
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
            'feature_mask_fn': tuple(
                modality.get_default_feature_mask_fn() for modality in self.modality
            ),
            'baseline_fn': tuple(
                modality.get_default_baseline_fn() for modality in self.modality
            ),
        }

    def _generate_default_kwargs_for_metric(self):
        return {
            'baseline_fn': tuple(
                modality.get_default_baseline_fn() for modality in self.modality
            ),
            'channel_dim': tuple(modality.channel_dim for modality in self.modality),
        }


class AutoExplanationForTSClassification(AutoExplanation):
    """
    An extension of AutoExplanation class with modality set to the TimeSeriesModality.

    Parameters:
        model (Model): The machine learning model to be analyzed.
        data (DataSource): The data source used for the experiment.
        input_extractor (Optional[Callable], optional): Custom function to extract input features.
        label_extractor (Optional[Callable], optional): Custom function to extract labels features.
        target_extractor (Optional[Callable], optional): Custom function to extract target features.
        target_labels (Optional[bool]): Whether to use target labels.
        sequence_dim (Tuple[int]): Sequence dimension.
        mask_agg_dim (Tuple[int]): A dimension for aggregating mask values. Usually, a channel dimension.

    Attributes:
        modality (TimeSeriesModality): An object to specify modality-specific workflow.
        recommended (RecommenderOutput): A data object, containing recommended explainers.
    """
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
