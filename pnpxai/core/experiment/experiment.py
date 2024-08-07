from typing import Any, Callable, Optional, Sequence, Union, List, Dict, Literal
import time
import warnings
import itertools

import torch
from torch import Tensor
import numpy as np
import optuna
from plotly import express as px
from plotly.graph_objects import Figure

from pnpxai.core._types import DataSource, Model, ModalityOrListOfModalities
from pnpxai.core.experiment.experiment_metrics_defaults import EVALUATION_METRIC_REVERSE_SORT, EVALUATION_METRIC_SORT_PRIORITY
from pnpxai.core.experiment.observable import ExperimentObservableEvent
from pnpxai.core.experiment.manager import ExperimentManager
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils.postprocess import PostProcessor
from pnpxai.evaluator.optimizer.objectives import Objective
from pnpxai.evaluator.optimizer.search_spaces import (
    create_default_search_space,
    create_default_suggest_methods,
)
from pnpxai.evaluator.optimizer.utils import (
    load_sampler,
    get_default_n_trials,
    format_params,
)
from pnpxai.evaluator.metrics.base import Metric
from pnpxai.messages import get_message

from pnpxai.utils import class_to_string, Observable, to_device, format_into_tuple

def default_input_extractor(x):
    return x[0]


def default_label_extractor(x):
    return x[-1]


def default_target_extractor(y):
    return y.argmax(-1)


class Experiment(Observable):
    """
    A class representing an experiment for model interpretability.

    Args:
        model (Model): The machine learning model to be interpreted.
        data (DataSource): The data used for the experiment.
        explainers (Sequence[Explainer]): Explainer objects or their arguments for interpreting the model.
        metrics (Optional[Sequence[Metric]]): Evaluation metrics used to assess model interpretability.
        modality (ModalityOrListOfModalities): The type of modality the model is designed for (default: "image").
        input_extractor (Optional[Callable[[Any], Any]]): Function to extract inputs from data (default: None).
        target_extractor (Optional[Callable[[Any], Any]]): Function to extract targets from data (default: None).
        input_visualizer (Optional[Callable[[Any], Any]]): Function to visualize input data (default: None).
        target_visualizer (Optional[Callable[[Any], Any]]): Function to visualize target data (default: None).

    Attributes:
        all_explainers (Sequence[Explainer]): All explainer objects used in the experiment.
        all_metrics (Sequence[Metric]): All evaluation metrics used in the experiment.
        errors (Sequence[Error]): 
        is_image_task (bool): True if the modality is an image-related modality, False otherwise.
        has_explanations (bool): True if the experiment has explanations, False otherwise.
    """

    def __init__(
        self,
        model: Model,
        data: DataSource,
        explainers: Sequence[Explainer],
        postprocessors: Optional[Sequence[Callable]] = None,
        metrics: Optional[Sequence[Metric]] = None,
        modality: ModalityOrListOfModalities = "image",
        input_extractor: Optional[Callable[[Any], Any]] = None,
        label_extractor: Optional[Callable[[Any], Any]] = None,
        target_extractor: Optional[Callable[[Any], Any]] = None,
        input_visualizer: Optional[Callable[[Any], Any]] = None,
        target_visualizer: Optional[Callable[[Any], Any]] = None,
        cache_device: Optional[Union[torch.device, str]] = None,
        target_labels: bool = False,
    ):
        super(Experiment, self).__init__()
        self.model = model
        self.model_device = next(self.model.parameters()).device

        self.manager = ExperimentManager(data=data, cache_device=cache_device)
        for explainer in explainers:
            self.manager.add_explainer(explainer)
        for postprocessor in postprocessors:
            self.manager.add_postprocessor(postprocessor)
        for metric in metrics:
            self.manager.add_metric(metric)

        self.input_extractor = input_extractor \
            if input_extractor is not None \
            else default_input_extractor
        self.label_extractor = label_extractor \
            if label_extractor is not None \
            else default_target_extractor
        self.target_extractor = target_extractor \
            if target_extractor is not None \
            else default_target_extractor
        self.input_visualizer = input_visualizer
        self.target_visualizer = target_visualizer
        self.modality = modality
        self.target_labels = target_labels
        self.reset_errors()

    def reset_errors(self):
        self._errors: List[BaseException] = []

    @property
    def errors(self):
        return self._errors


    def to_device(self, x):
        return to_device(x, self.model_device)

    def run_batch(
        self,
        data_ids: List[int],
        explainer_id: int,
        postprocessor_id: int,
        metric_id: int,
    ):
        self.predict_batch(data_ids)
        self.explain_batch(data_ids, explainer_id)
        self.evaluate_batch(
            data_ids, explainer_id, postprocessor_id, metric_id)
        data = self.manager.batch_data_by_ids(data_ids)
        return {
            'inputs': self.input_extractor(data),
            'labels': self.label_extractor(data),
            'outputs': self.manager.batch_outputs_by_ids(data_ids),
            'targets': self._get_targets(data_ids),
            'explainer': self.manager.get_explainer_by_id(explainer_id),
            'explanation': self.manager.batch_explanations_by_ids(data_ids, explainer_id),
            'postprocessor': self.manager.get_postprocessor_by_id(postprocessor_id),
            'postprocessed': self.postprocess_batch(data_ids, explainer_id, postprocessor_id),
            'metric': self.manager.get_metric_by_id(metric_id),
            'evaluation': self.manager.batch_evaluations_by_ids(data_ids, explainer_id, postprocessor_id, metric_id),
        }

    def predict_batch(
        self,
        data_ids: List[int],
    ):
        data_ids_pred = [
            idx for idx in data_ids
            if self.manager.get_output_by_id(idx) is None
        ]
        if len(data_ids_pred) > 0:
            data = self.manager.batch_data_by_ids(data_ids_pred)
            outputs = self.model(*format_into_tuple(self.input_extractor(data)))
            self.manager.cache_outputs(data_ids_pred, outputs)
        return self.manager.batch_outputs_by_ids(data_ids)

    def explain_batch(
        self,
        data_ids: List[int],
        explainer_id: int,
    ):
        data_ids_expl = [
            idx for idx in data_ids
            if self.manager.get_explanation_by_id(idx, explainer_id) is None
        ]
        if len(data_ids_expl):
            data = self.manager.batch_data_by_ids(data_ids_expl)
            inputs = self.input_extractor(data)
            targets = self._get_targets(data_ids_expl)
            explainer = self.manager.get_explainer_by_id(explainer_id)
            explanations = explainer.attribute(inputs, targets)
            self.manager.cache_explanations(
                explainer_id, data_ids_expl, explanations)
        return self.manager.batch_explanations_by_ids(data_ids, explainer_id)

    def postprocess_batch(
        self,
        data_ids: List[int],
        explainer_id: int,
        postprocessor_id: int,
    ):
        explanations = self.manager.batch_explanations_by_ids(data_ids, explainer_id)
        postprocessor = self.manager.get_postprocessor_by_id(postprocessor_id)
        return postprocessor(explanations)

    def evaluate_batch(
        self,
        data_ids: List[int],
        explainer_id: int,
        postprocessor_id: int,
        metric_id: int,
    ):
        data_ids_eval = [
            idx for idx in data_ids
            if self.manager.get_evaluation_by_id(
                idx, explainer_id, postprocessor_id, metric_id) is None
        ]
        if len(data_ids_eval):
            data = self.manager.batch_data_by_ids(data_ids_eval)
            inputs = self.input_extractor(data)
            targets = self._get_targets(data_ids_eval)
            postprocessed = self.postprocess_batch(
                data_ids_eval, explainer_id, postprocessor_id)
            explainer = self.manager.get_explainer_by_id(explainer_id)
            metric = self.manager.get_metric_by_id(metric_id)
            evaluations = metric.set_explainer(explainer).evaluate(
                inputs, targets, postprocessed)
            self.manager.cache_evaluations(
                explainer_id, postprocessor_id, metric_id,
                data_ids_eval, evaluations
            )
        return self.manager.batch_evaluations_by_ids(
            data_ids, explainer_id, postprocessor_id, metric_id
        )


    def _get_targets(self, data_ids):
        if self.target_labels:
            return self.label_extractor(self.manager.batch_data_by_ids(data_ids))
        outputs = self.manager.batch_outputs_by_ids(data_ids)
        return self.target_extractor(outputs)

    def optimize(
        self,
        data_id: int,
        explainer_id: int,
        metric_id: int,
        direction: Literal['minimize', 'maximize']='maximize',
        sampler: Literal['grid', 'random', 'tpe']='tpe',
        suggest_methods: Optional[Dict[str, Callable]]=None,
        search_space: Optional[Dict[str, Any]]=None,
        n_trials: Optional[int]=None,
        timeout: Optional[float]=None,
        return_study: bool=False,
        **kwargs, # sampler kwargs
    ):
        assert n_trials is None or timeout is None
        data = self.manager.batch_data_by_ids([data_id])
        explainer = self.manager.get_explainer_by_id(explainer_id)
        postprocessor = self.manager.get_postprocessor_by_id(0) # sample postprocessor to ensure channel_dim
        metric = self.manager.get_metric_by_id(metric_id)
        suggest_methods = suggest_methods or create_default_suggest_methods(explainer)
        objective = Objective(
            explainer=explainer,
            metric=metric,
            suggest_methods=suggest_methods,
            channel_dim=postprocessor.channel_dim,
            inputs=self.input_extractor(data),
            targets=self._get_targets([data_id])
        )
        if sampler == 'grid':
            kwargs['search_space'] = search_space or create_default_search_space(explainer)
        if timeout is None:
            n_trials = n_trials or get_default_n_trials(sampler)

        # optimize
        study = optuna.create_study(
            sampler=load_sampler(sampler, **kwargs),
            direction=direction,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
        )

        # update explainer
        best_params = study.best_params.copy()
        explainer_kwargs = format_params(best_params)
        postprocessor_kwargs = {
            'pooling_method': explainer_kwargs.pop('pooling_method', 'sumpos'),
            'normalization_method': explainer_kwargs.pop('normalization_method', 'minmax'),
            'channel_dim': objective.channel_dim,
        }
        opt_explainer = explainer.set_kwargs(**explainer_kwargs)
        opt_postprocessor = PostProcessor(**postprocessor_kwargs)
        opt_explainer_id = self.manager.add_explainer(opt_explainer)
        opt_postprocessor_id = self.manager.add_postprocessor(opt_postprocessor) # TODO: find same postprocessor
        if not return_study:
            return opt_explainer_id, opt_postprocessor_id
        return opt_explainer_id, opt_postprocessor_id, study

    def run(
        self,
        data_ids: Optional[Sequence[int]] = None,
        explainer_ids: Optional[Sequence[int]] = None,
        postprocessor_ids: Optional[Sequence[int]] = None,
        metric_ids: Optional[Sequence[int]] = None,
    ) -> 'Experiment':
        """
        Run the experiment by processing data, generating explanations, evaluating with metrics, caching and retrieving the data.

        Args:
            data_ids (Optional[Sequence[int]]): A sequence of data IDs to specify the subset of data to process.
            explainer_ids (Optional[Sequence[int]]): A sequence of explainer IDs to specify the subset of explainers to use.
            metrics_ids (Optional[Sequence[int]]): A sequence of metric IDs to specify the subset of metrics to evaluate.

        Returns:
            The Experiment instance with updated results and state.

        This method orchestrates the experiment by configuring the manager, obtaining explainer and metric instances,
        processing data, generating explanations, and evaluating metrics. It then saves the results in the manager.

        Note: The input parameters allow for flexibility in specifying subsets of data, explainers, and metrics to process.
        If not provided, the method processes all available data, explainers, and metrics.
        """
        self.reset_errors()
        # self.manager.set_config(data_ids, explainer_ids, metrics_ids)
        
        # inference

        # data_ids is filtered out data indices whose output is saved in cache
        data, data_ids_pred = self.manager.get_data_to_predict(data_ids)
        outputs = self._predict(data)
        self.manager.save_outputs(outputs, data, data_ids_pred)

        # explain
        explainers, _ = self.manager.get_explainers(explainer_ids)
        postprocessors, _ = self.manager.get_postprocessors(postprocessor_ids)
        metrics, _ = self.manager.get_metrics(metric_ids)

        for explainer, explainer_id in zip(explainers, explainer_ids):
            explainer_name = class_to_string(explainer)

            # data_ids is filtered out data indices whose explanation is saved in cache
            data, data_ids_expl = self.manager.get_data_to_process_for_explainer(
                explainer_id, data_ids)
            explanations = self._explain(data, data_ids_expl, explainer)
            self.manager.save_explanations(
                explanations, data, data_ids_expl, explainer_id
            )
            message = get_message(
                'experiment.event.explainer', explainer=explainer_name
            )
            print(f"[Experiment] {message}")
            self.fire(ExperimentObservableEvent(
                self.manager, message, explainer))

            for postprocessor, postprocessor_id in zip(postprocessors, postprocessor_ids):
                for metric, metric_id in zip(metrics, metric_ids):
                    metric_name = class_to_string(metric)
                    data, data_ids_eval = self.manager.get_data_to_process_for_metric(
                        explainer_id, postprocessor_id, metric_id, data_ids)
                    explanations, data_ids_eval = self.manager.get_valid_explanations(
                        explainer_id, data_ids_eval)
                    data, _ = self.manager.get_data(data_ids_eval)
                    evaluations = self._evaluate(
                        data, data_ids_eval, explanations, explainer, postprocessor, metric)
                    self.manager.save_evaluations(
                        evaluations, data, data_ids_eval, explainer_id, postprocessor_id, metric_id)

                    message = get_message(
                        'experiment.event.explainer.metric', explainer=explainer_name, metric=metric_name)
                    print(f"[Experiment] {message}")
                    self.fire(ExperimentObservableEvent(
                        self.manager, message, explainer, metric))
        return self

    def _predict(self, data: DataSource):
        outputs = [
            self.model(*format_into_tuple(
                self.to_device(self.input_extractor(datum)))
            ) for datum in data
        ]
        return outputs

    def _explain(self, data: DataSource, data_ids: List[int], explainer: Explainer):
        explanations = [None] * len(data)
        explainer_name = class_to_string(explainer)
        for i, (datum, data_id) in enumerate(zip(data, data_ids)):
            try:
                datum = self.to_device(datum)
                inputs = format_into_tuple(self.input_extractor(datum))
                targets = self.label_extractor(datum) if self.target_labels \
                    else self._get_targets(data_ids)
                explanations[i] = explainer.attribute(
                    inputs=inputs,
                    targets=targets,
                )
            except NotImplementedError as error:
                warnings.warn(
                    f"\n[Experiment] {get_message('experiment.errors.explainer_unsupported', explainer=explainer_name)}")
                raise error
            except Exception as e:
                warnings.warn(
                    f"\n[Experiment] {get_message('experiment.errors.explanation', explainer=explainer_name, error=e)}")
                self._errors.append(e)
        return explanations

    # def _evaluate(self, data: DataSource, data_ids: List[int], explanations: DataSource, explainer: Explainer, postprocessor: Callable, metric: Metric):
    #     if explanations is None:
    #         return None
    #     started_at = time.time()
    #     metric_name = class_to_string(metric)
    #     explainer_name = class_to_string(explainer)

    #     evaluations = [None] * len(data)
    #     for i, (datum, data_id, explanation) in enumerate(zip(data, data_ids, explanations)):
    #         if explanation is None:
    #             continue
    #         datum = self.to_device(datum)
    #         explanation = self.to_device(explanation)
    #         inputs = self.input_extractor(datum)
    #         targets = self.label_extractor(datum) if self.target_labels \
    #             else self._get_targets(data_ids)
    #         try:
    #             metric = metric.set_explainer(explainer).set_postprocessor(postprocessor)
    #             evaluations[i] = metric.evaluate(
    #                 inputs=inputs,
    #                 targets=targets,
    #                 attributions=explanation,
    #             )
    #         except Exception as e:
    #             import pdb; pdb.set_trace()
    #             warnings.warn(
    #                 f"\n[Experiment] {get_message('experiment.errors.evaluation', explainer=explainer_name, metric=metric_name, error=e)}")
    #             self._errors.append(e)
    #     elapsed_time = time.time() - started_at
    #     print(
    #         f"[Experiment] {get_message('elapsed', modality=metric_name, elapsed=elapsed_time)}")

    #     return evaluations

    def _get_targets(self, data_ids: List[int]):
        # predict if not cached
        not_predicted = [
            data_id for data_id in data_ids
            if self.manager._cache.get_output(data_id) is None
        ]
        if len(not_predicted) > 0:
            self.predict_batch(not_predicted)

        # load outputs from cache
        outputs = torch.stack([
            self.manager._cache.get_output(data_id)
            for data_id in data_ids
        ])
        return self.target_extractor(outputs)

    # def get_visualizations_flattened(self) -> Sequence[Sequence[Figure]]:
    #     """
    #     Generate flattened visualizations for each data point based on explanations.

    #     Returns:
    #         List of visualizations for each data point across explainers.

    #     This method retrieves valid explanations for each explainer, formats them for visualization,
    #     and generates visualizations using Plotly Express. The results are flattened based on the data points
    #     and returned as a list of lists of figures.
    #     """
    #     assert self.modality == 'image', f"Visualization for '{self.modality} is not supported yet"
    #     explainers, explainer_ids = self.manager.get_explainers()
    #     # Get all data ids
    #     experiment_data_ids = self.manager.get_data_ids()
    #     visualizations = []

    #     for explainer, explainer_id in zip(explainers, explainer_ids):
    #         # Get all valid explanations and data ids for this explainer
    #         explanations, data_ids = self.manager.get_valid_explanations(
    #             explainer_id)
    #         data = self.manager.get_data(data_ids)[0]
    #         explainer_visualizations = []
    #         for explanation in explanations:
    #             figs = []
    #             for attr in explanation:
    #                 postprocessed = postprocess_attr(
    #                     attr,
    #                     channel_dim=0,
    #                     pooling_method='l2normsq',
    #                     normalization_method='minmax'
    #                 ).detach().numpy()
    #                 fig = px.imshow(postprocessed, color_continuous_scale='RdBu_R', color_continuous_midpoint=.5)
    #                 figs.append(fig)
    #             explainer_visualizations.append(figs)

    #         # explainer_visualizations = [[
    #         #     px.imshow(
    #         #         postprocess_attr(
    #         #             attr, # C x H x W
    #         #             channel_dim=0,
    #         #             pooling_method='l2normsq',
    #         #             normalization_method='minmax',
    #         #         ).detach().numpy(),
    #         #         color_continuous_scale='RdBu_R',
    #         #         color_continuous_midpoint=.5,
    #         #     ) for attr in explanation]
    #         #     for explanation in explanations
    #         # ]
    #         # # Visualize each valid explanataion
    #         # for datum, explanation in zip(data, explanations):
    #         #     inputs = self.input_extractor(datum)
    #         #     targets = self.target_extractor(datum)
    #         #     formatted = explainer.format_outputs_for_visualization(
    #         #         inputs=inputs,
    #         #         targets=targets,
    #         #         explanations=explanation,
    #         #         modality=self.modality
    #         #     )

    #         #     if not self.manager.is_batched:
    #         #         formatted = [formatted]
    #         #     formatted_visualizations = [
    #         #         px.imshow(explanation, color_continuous_scale="RdBu_r", color_continuous_midpoint=0.0) for explanation in formatted
    #         #     ]
    #         #     if not self.manager.is_batched:
    #         #         formatted_visualizations = formatted_visualizations[0]
    #         #     explainer_visualizations.append(formatted_visualizations)

    #         flat_explainer_visualizations = self.manager.flatten_if_batched(
    #             explainer_visualizations, data)
    #         # Set visualizaions of all data ids as None
    #         explainer_visualizations = {
    #             idx: None for idx in experiment_data_ids}
    #         # Fill all valid visualizations
    #         for visualization, data_id in zip(flat_explainer_visualizations, data_ids):
    #             explainer_visualizations[data_id] = visualization
    #         visualizations.append(list(explainer_visualizations.values()))

    #     return visualizations

    def get_inputs_flattened(self, data_ids: Optional[List[int]]=None) -> Sequence[Tensor]:
        """
        Retrieve and flatten last run input data.

        Returns:
            Flattened input data.

        This method retrieves input data using the input extractor and flattens it for further processing.
        """
        data, _ = self.manager.get_data(data_ids)
        data = [self.input_extractor(datum) for datum in data]
        return self.manager.flatten_if_batched(data, data)

    def get_all_inputs_flattened(self) -> Sequence[Tensor]:
        """
        Retrieve and flatten all input data.

        Returns:
            Flattened input data from all available data.

        This method retrieves input data from all available data points using the input extractor and flattens it.
        """
        data = self.manager.get_all_data()
        data = [self.input_extractor(datum) for datum in data]
        return self.manager.flatten_if_batched(data, data)

    def get_labels_flattened(self, data_ids: Optional[List[int]]=None) -> Sequence[Tensor]:
        data, _ = self.manager.get_data(data_ids)
        labels = [self.label_extractor(datum) for datum in data]
        return self.manager.flatten_if_batched(labels, data)

    def get_targets_flattened(self, data_ids: Optional[List[int]]=None) -> Sequence[Tensor]:
        """
        Retrieve and flatten target data.

        Returns:
            Flattened target data.

        This method retrieves target data using the target extractor and flattens it for further processing.
        """
        if self.target_labels:
            return self.get_labels_flattened(data_ids)
        data, _ = self.manager.get_data(data_ids)
        targets = [self._get_targets(data_ids)]
        return self.manager.flatten_if_batched(targets, data)
        # targets = [self.label_extractor(datum) for datum in data] \
        #     if self.target_labels else [self._get_targets(data_ids)]
        # return self.manager.flatten_if_batched(targets, data)

    def get_outputs_flattened(self, data_ids: Optional[List[int]]=None) -> Sequence[Tensor]:
        """
        Retrieve and flatten model outputs.

        Returns:
            Flattened model outputs.

        This method retrieves flattened model outputs using the manager's get_flat_outputs method.
        """
        return self.manager.get_flat_outputs()

    def get_explanations_flattened(self, data_ids: Optional[List[int]]=None) -> Sequence[Sequence[Tensor]]:
        """
        Retrieve and flatten explanations from all explainers.

        Returns:
            Flattened explanations from all explainers.

        This method retrieves flattened explanations for each explainer using the manager's get_flat_explanations method.
        """
        _, explainer_ids = self.manager.get_explainers()
        return [
            self.manager.get_flat_explanations(explainer_id, data_ids)
            for explainer_id in explainer_ids
        ]

    def get_evaluations_flattened(self, data_ids: Optional[List[int]]=None) -> Sequence[Sequence[Sequence[Tensor]]]:
        """
        Retrieve and flatten evaluations for all explainers and metrics.

        Returns:
            Flattened evaluations for all explainers and metrics.

        This method retrieves flattened evaluations for each explainer and metric using the manager's
        get_flat_evaluations method.
        """
        _, explainer_ids = self.manager.get_explainers()
        _, postprocessor_ids = self.manager.get_postprocessors()
        _, metric_ids = self.manager.get_metrics()

        formatted = [[[
                self.manager.get_flat_evaluations(
                    explainer_id, postprocessor_id, metric_id, data_ids)
                for metric_id in metric_ids
            ] for postprocessor_id in postprocessor_ids]
        for explainer_id in explainer_ids]

        return formatted

    def get_explainers_ranks(self) -> Optional[Sequence[Sequence[int]]]:
        """
        Calculate and return rankings for explainers based on evaluations.

        Returns:
            Rankings of explainers. Returns None if rankings cannot be calculated.

        This method calculates rankings for explainers based on evaluations and metric scores. It considers
        metric priorities and sorting preferences to produce rankings.
        """
        evaluations = [[[
            data_evaluation.detach().cpu() if data_evaluation is not None else None
            for data_evaluation in metric_data
        ]for metric_data in explainer_data
        ]for explainer_data in self.get_evaluations_flattened()]
        # (explainers, metrics, data)
        evaluations = np.array(evaluations, dtype=float)
        evaluations = np.nan_to_num(evaluations, nan=-np.inf)
        if evaluations.ndim != 3:
            return None

        evaluations = evaluations.argsort(axis=-3).argsort(axis=-3) + 1
        n_explainers = evaluations.shape[0]
        metric_name_to_idx = {}

        for idx, metric in enumerate(self.get_current_metrics()):
            metric_name = class_to_string(metric)
            evaluations[:, idx, :] = evaluations[:, idx, :]
            if EVALUATION_METRIC_REVERSE_SORT.get(metric_name, False):
                evaluations[:, idx, :] = \
                    n_explainers - evaluations[:, idx, :] + 1

            metric_name_to_idx[metric_name] = idx
        # (explainers, data)
        scores: np.ndarray = evaluations.sum(axis=-2)

        for metric_name in EVALUATION_METRIC_SORT_PRIORITY:
            if metric_name not in metric_name_to_idx:
                continue

            idx = metric_name_to_idx[metric_name]
            scores = scores * n_explainers + evaluations[:, idx, :]

        return scores.argsort(axis=-2).argsort(axis=-2).tolist()

    # @property
    # def is_image_task(self):
    #     return self.modality == 'image'

    @property
    def has_explanations(self):
        return self.manager.has_explanations
