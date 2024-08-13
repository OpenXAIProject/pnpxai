from abc import abstractmethod
from typing import Literal, Callable, Optional, Type, Union, Tuple, List

from torch.nn.modules import Module
from torch.utils.data import DataLoader

from pnpxai.core.experiment.experiment import Experiment
from pnpxai.core.detector import detect_model_architecture
from pnpxai.core.recommender import XaiRecommender
from pnpxai.core._types import DataSource, Model, ModalityOrTupleOfModalities, Modality
from pnpxai.explainers.types import TargetLayer, ForwardArgumentExtractor
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
        input_extractor: Optional[Callable]=None,
        label_extractor: Optional[Callable]=None,
        target_extractor: Optional[Callable]=None,
        target_labels: bool=False,
        channel_dim: Union[int, Tuple[int]]=1,
    ):
        self.channel_dim = channel_dim
        super().__init__(
            model=model,
            data=data,
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

    def _load_default_postprocessors(self):
        return all_postprocessors(self.channel_dim)

    def _load_default_metrics(self, model):
        empty_metrics = [] # empty means that explainer is not assigned yet
        for metric_type in DEFAULT_METRICS:
            metric = metric_type(model=model)
            default_kwargs = self._generate_default_kwargs_for_metric()
            for k, v in default_kwargs.items():
                if hasattr(metric, k):
                    metric = metric.set_kwargs(**{k: v})
            empty_metrics.append(metric)
        return empty_metrics

    @abstractmethod
    def _generate_default_kwargs_for_explainer(self):
        raise NotImplementedError

    @abstractmethod
    def _generate_default_kwargs_for_metric(self):
        raise NotImplementedError


class AutoExplanationForImageClassification(AutoExplanation):
    def __init__(
        self,
        model: Module,
        data: DataLoader,
        input_extractor: Optional[Callable]=None,
        label_extractor: Optional[Callable]=None,
        target_extractor: Optional[Callable]=None,
        target_labels: bool=False,
        channel_dim: int=1,
    ):
        self.recommended = XaiRecommender().recommend(modality='image', model=model)
        super().__init__(
            model=model,
            data=data,
            input_extractor=input_extractor,
            label_extractor=label_extractor,
            target_extractor=target_extractor,
            target_labels=target_labels,
            channel_dim=channel_dim,
        )

    def _generate_default_kwargs_for_explainer(self):
        return {
            'feature_mask_fn': get_default_feature_mask_fn(modality='image'),
            'baseline_fn': get_default_baseline_function(modality='image'),
        }

    def _generate_default_kwargs_for_metric(self):
        return {
            'baseline_fn': get_default_baseline_function(modality='image'),
            'channel_dim': self.channel_dim or get_default_channel_dim(modality='image'),
        }


class AutoExplanationForTextClassification(AutoExplanation):
    def __init__(
        self,
        model: Module,
        data: DataLoader,
        layer: TargetLayer,
        mask_token_id: int,
        input_extractor: Optional[Callable]=None,
        forward_arg_extractor: Optional[Callable]=None,
        additional_forward_arg_extractor: Optional[Callable]=None,
        label_extractor: Optional[Callable]=None,
        target_extractor: Optional[Callable]=None,
        target_labels: bool=False,
        channel_dim: int=-1,
    ):
        self.layer = layer
        self.mask_token_id = mask_token_id
        self.forward_arg_extractor = forward_arg_extractor
        self.additional_forward_arg_extractor = additional_forward_arg_extractor

        self.recommended = XaiRecommender().recommend(modality='text', model=model)
        super().__init__(
            model=model,
            data=data,
            input_extractor=input_extractor,
            label_extractor=label_extractor,
            target_extractor=target_extractor,
            target_labels=target_labels,
            channel_dim=channel_dim,
        )

    def _generate_default_kwargs_for_explainer(self):
        return {
            'layer': self.layer,
            'forward_arg_extractor': self.forward_arg_extractor,
            'additional_forward_arg_extractor': self.additional_forward_arg_extractor,
            'feature_mask_fn': get_default_feature_mask_fn(modality='text'),
            'baseline_fn': get_default_baseline_function(
                modality='text',
                mask_token_id=self.mask_token_id,
            ),
        }

    def _generate_default_kwargs_for_metric(self):
        return {
            'baseline_fn': get_default_baseline_function(
                modality='text',
                mask_token_id=self.mask_token_id,
            ),
            'channel_dim': self.channel_dim or get_default_channel_dim(modality='text'),
        }


class AutoExplanationForVisualQuestionAnswering(AutoExplanation):
    def __init__(
        self,
        model: Module,
        data: DataLoader,
        layer: List[TargetLayer],
        mask_token_id: int,
        input_extractor: Optional[Callable]=None,
        forward_arg_extractor: Optional[Callable]=None,
        additional_forward_arg_extractor: Optional[Callable]=None,
        label_extractor: Optional[Callable]=None,
        target_extractor: Optional[Callable]=None,
        target_labels: bool=False,
        channel_dim: Tuple[int]=(1, -1),
    ):
        self.layer = layer
        self.mask_token_id = mask_token_id
        self.forward_arg_extractor = forward_arg_extractor
        self.additional_forward_arg_extractor = additional_forward_arg_extractor

        self.recommended = XaiRecommender().recommend(
            modality=('image', 'text'),
            model=model,
        )
        super().__init__(
            model=model,
            data=data,
            input_extractor=input_extractor,
            label_extractor=label_extractor,
            target_extractor=target_extractor,
            target_labels=target_labels,
            channel_dim=channel_dim,
        )

    def _generate_default_kwargs_for_explainer(self):
        return {
            'layer': self.layer,
            'forward_arg_extractor': self.forward_arg_extractor,
            'additional_forward_arg_extractor': self.additional_forward_arg_extractor,
            'feature_mask_fn': get_default_feature_mask_fn(modality=('image', 'text')),
            'baseline_fn': get_default_baseline_function(
                modality=('image', 'text'),
                mask_token_id=self.mask_token_id,
            ),
        }

    def _generate_default_kwargs_for_metric(self):
        return {
            'baseline_fn': get_default_baseline_function(
                modality='text',
                mask_token_id=self.mask_token_id,
            ),
            'channel_dim': self.channel_dim \
                or get_default_channel_dim(modality=('image', 'text')),
        }


class AutoExplanationForTabularClassification(AutoExplanation):
    def __init__(self):
        pass

    def _check_background_data(self, kwargs):
        if self.modality == 'tabular':
            assert kwargs.get('background_data') is not None, "Must have 'background_data' for tabular modality."
