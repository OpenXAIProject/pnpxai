from typing import Literal, Callable, Optional, Type

from pnpxai.core.experiment.experiment import Experiment
from pnpxai.core.detector import detect_model_architecture
from pnpxai.core.recommender import XaiRecommender
from pnpxai.core._types import DataSource, Model, ModalityOrListOfModalities, Question, Modality
from pnpxai.explainers import CAM_BASED_EXPLAINERS, ATTENTION_SPECIFIC_EXPLAINERS, PERTURBATION_BASED_EXPLAINERS
from pnpxai.explainers import (
    CAM_BASED_EXPLAINERS,
    PERTURBATION_BASED_EXPLAINERS,
    GRADIENT_BASED_EXPLAINERS,
    EXPLAINERS_FOR_TABULAR,
    IntegratedGradients,
)
from pnpxai.explainers.types import TargetLayerOrListOfTargetLayers, ForwardArgumentExtractor
from pnpxai.explainers.utils import get_default_feature_mask_fn, get_default_baseline_fn


EXPLAINERS_NO_KWARGS = CAM_BASED_EXPLAINERS
EXPLAINERS_FEATURE_MASK_AVAILABLE = PERTURBATION_BASED_EXPLAINERS
EXPLAINERS_BASELINE_FN_AVAILABLE = PERTURBATION_BASED_EXPLAINERS + [IntegratedGradients]
EXPLAINERS_LAYER_AVAILABLE = GRADIENT_BASED_EXPLAINERS
EXPLAINERS_BG_DATA_REQUIRED = EXPLAINERS_FOR_TABULAR


class AutoExperiment(Experiment):
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
        modality: ModalityOrListOfModalities = "image",
        question: Question = "why",
        evaluator_enabled: bool = True,
        input_extractor: Optional[Callable] = None,
        label_extractor: Optional[Callable] = None,
        target_extractor: Optional[Callable] = None,
        input_visualizer: Optional[Callable] = None,
        target_visualizer: Optional[Callable] = None,
        target_labels: bool = False,
        **kwargs,
    ):
        if (modality == 'text' or 'text' in modality):
            assert kwargs.get('layer'), "Must have 'layer' for text modality. It might be a word embedding layer of your model."
        if modality == 'tabular':
            assert kwargs.get('background_data') is not None, "Must have 'background_data' for tabular modality."

        # recommender
        self.recommended = XaiRecommender().recommend(modality=modality, model=model)

        # explainers
        explainers = []
        for explainer_type in self.recommended.explainers:
            explainer_kwargs = {}
            if explainer_type in EXPLAINERS_BG_DATA_REQUIRED:
                explainer_kwargs['background_data'] = kwargs.get('background_data')
            elif explainer_type not in EXPLAINERS_NO_KWARGS:
                explainer_kwargs['forward_arg_extractor'] = kwargs.get('forward_arg_extractor')
                explainer_kwargs['additional_forward_arg_extractor'] = kwargs.get('additional_forward_arg_extractor')
                if explainer_type in EXPLAINERS_FEATURE_MASK_AVAILABLE:
                    explainer_kwargs['feature_mask_fn'] = kwargs.get('feature_mask_fn') \
                        or get_default_feature_mask_fn(modality)
                if explainer_type in EXPLAINERS_BASELINE_FN_AVAILABLE:
                    explainer_kwargs['baseline_fn'] = kwargs.get('baseline_fn') \
                        or get_default_baseline_fn(modality, mask_token_id=kwargs.get('mask_token_id') or 0)
                if explainer_type in EXPLAINERS_LAYER_AVAILABLE:
                    explainer_kwargs['layer'] = kwargs.get('layer')
            explainers.append(explainer_type(model, **explainer_kwargs))

        # # TODO: metrics for other modalities
        if modality != 'tabular' and evaluator_enabled:
            raise NotImplementedError(f"Evaluator for {modality} is not supported yet.")
        
        metrics = [
            metric_type(model)
            for metric_type in self.recommended.metrics
        ] if evaluator_enabled else []
        super().__init__(
            model=model,
            data=data,
            explainers=explainers,
            metrics=metrics,
            modality=modality,
            input_extractor=input_extractor,
            label_extractor=label_extractor,
            target_extractor=target_extractor,
            input_visualizer=input_visualizer,
            target_visualizer=target_visualizer,
            target_labels=target_labels,
        )

