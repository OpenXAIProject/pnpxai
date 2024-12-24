from typing import Any, Callable, Optional, Sequence, Union, List, Literal

import torch
from torch import Tensor
import numpy as np
import optuna

from pnpxai.core._types import DataSource, Model
from pnpxai.core.modality.modality import Modality, TextModality
from pnpxai.core.experiment.experiment_metrics_defaults import EVALUATION_METRIC_REVERSE_SORT, EVALUATION_METRIC_SORT_PRIORITY
from pnpxai.core.experiment.manager import ExperimentManager
from pnpxai.explainers import Explainer, Lime, KernelShap
from pnpxai.explainers.utils.postprocess import Identity
from pnpxai.evaluator.optimizer.types import OptimizationOutput
from pnpxai.evaluator.optimizer.objectives import Objective
from pnpxai.evaluator.optimizer.utils import (
    load_sampler,
    get_default_n_trials,
)
from pnpxai.evaluator.metrics.base import Metric

from pnpxai.utils import (
    class_to_string, Observable, to_device,
    format_into_tuple, format_out_tuple_if_single,
)


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
        modality (Modality): The type of modality (image, tabular, text, time series) the model is designed for.
        explainers (Sequence[Explainer]): Explainer objects or their arguments for interpreting the model.
        postprocessors (Optional[Sequence[Callable]]): Postprocessing functions to apply to explanations.
        metrics (Optional[Sequence[Metric]]): Evaluation metrics used to assess model interpretability.
        input_extractor (Optional[Callable[[Any], Any]]): Function to extract inputs from data.
        label_extractor (Optional[Callable[[Any], Any]]): Function to extract labels from data.
        target_extractor (Optional[Callable[[Any], Any]]): Function to extract targets from data.
        cache_device (Optional[Union[torch.device, str]]): Device to cache data and results.
        target_labels (bool): True if the target is a label, False otherwise.

    Attributes:
        modality (Modality): Object defining the modality-specific control flow of the experiment.
        manager (ExperimentManager): Manager object for the experiment.
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
        modality: Modality,
        explainers: Sequence[Explainer],
        postprocessors: Sequence[Callable],
        metrics: Sequence[Metric],
        input_extractor: Optional[Callable[[Any], Any]] = None,
        label_extractor: Optional[Callable[[Any], Any]] = None,
        target_extractor: Optional[Callable[[Any], Any]] = None,
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
        self.target_labels = target_labels
        self.modality = modality
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
        explainer_id: int,
        postprocessor_id: int,
        metric_id: int,
        data_ids: Optional[Sequence[int]] = None,
    ) -> dict:
        """
        Runs the experiment for selected batch of data, explainer, postprocessor and metric.

        Args:
            data_ids (Sequence[int]): A sequence of data IDs to specify the subset of data to process.
            explainer_id (int): ID of explainer to use for the run.
            postprocessor_id (int): ID of postprocessor to use for the run.
            metrics_id (int): ID of metric to use for the run.

        Returns:
            The dictionary of inputs, labels, outputs, targets, explainer, explanation, postprocessor, postprocessed, metric, and evaluation.

        This method orchestrates the experiment by configuring the manager, obtaining explainer and metric instances,
        processing data, generating explanations, and evaluating metrics. It then caches the results in the manager, and returns back to the user.

        Note: The input parameters allow for flexibility in specifying subset of data, explainer, postprocessor and metric to process.
        """
        data_ids = data_ids if data_ids is not None else self.manager.get_data_ids(
            data_ids)

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
        data_ids: Sequence[int],
    ):
        """
        Predicts results of the experiment for selected batch of data.

        Args:
            data_ids (Sequence[int]): A sequence of data IDs to specify the subset of data to process.

        Returns:
            Batched model outputs corresponding to data ids.

        This method orchestrates the experiment by configuring the manager and processing data. It then caches the results in the manager, and returns back to the user.
        """

        data_ids_pred = [
            idx for idx in data_ids
            if self.manager.get_output_by_id(idx) is None
        ]
        if len(data_ids_pred) > 0:
            data = self.manager.batch_data_by_ids(data_ids_pred)
            outputs = self.model(
                *format_into_tuple(self.input_extractor(data)))
            self.manager.cache_outputs(data_ids_pred, outputs)
        return self.manager.batch_outputs_by_ids(data_ids)

    def explain_batch(
        self,
        data_ids: Sequence[int],
        explainer_id: int,
    ):
        """
        Explains selected batch of data within experiment.

        Args:
            data_ids (Sequence[int]): A sequence of data IDs to specify the subset of data to process.

        Returns:
            Batched model explanations corresponding to data ids.

        This method orchestrates the experiment by configuring the manager, obtaining explainer instance,
        processing data, and generating explanations. It then caches the results in the manager, and returns back to the user.
        """
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
        """
        Postprocesses selected batch of data within experiment.

        Args:
            data_ids (Sequence[int]): A sequence of data IDs to specify the subset of data to postprocess.
            explainer_id (int): An explainer ID to specify the explainer to use.
            postprocessor_id (int): A postprocessor ID to specify the postprocessor to use.

        Returns:
            Batched postprocessed model explanations corresponding to data ids.

        This method orchestrates the experiment by configuring the manager, obtaining explainer instance,
        processing data, and generating explanations. It then caches the results in the manager, and returns back to the user.
        """
        explanations = self.manager.batch_explanations_by_ids(
            data_ids, explainer_id)
        postprocessor = self.manager.get_postprocessor_by_id(postprocessor_id)

        modalities = format_into_tuple(self.modality)
        explanations = format_into_tuple(explanations)
        postprocessors = format_into_tuple(postprocessor)

        batch = []
        explainer = self.manager.get_explainer_by_id(explainer_id)
        for mod, attr, pp in zip(modalities, explanations, postprocessors):
            if (
                isinstance(explainer, (Lime, KernelShap))
                and isinstance(mod, TextModality)
                and not isinstance(pp.pooling_fn, Identity)
            ):
                raise ValueError(
                    f'postprocessor {postprocessor_id} does not support explainer {explainer_id}.')
            batch.append(pp(attr))
        return format_out_tuple_if_single(batch)

    def evaluate_batch(
        self,
        data_ids: List[int],
        explainer_id: int,
        postprocessor_id: int,
        metric_id: int,
    ):
        """
        Evaluates selected batch of data within experiment.

        Args:
            data_ids (Sequence[int]): A sequence of data IDs to specify the subset of data to postprocess.
            explainer_id (int): An explainer ID to specify the explainer to use.
            postprocessor_id (int): A postprocessor ID to specify the postprocessor to use.
            metric_id (int): A metric ID to evaluate the model explanations.

        Returns:
            Batched model evaluations corresponding to data ids.

        This method orchestrates the experiment by configuring the manager, obtaining explainer instance,
        processing data, generating explanations, and evaluating results. It then caches the results in the manager, and returns back to the user.
        """
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
        data_ids: Union[int, Sequence[int]],
        explainer_id: int,
        metric_id: int,
        direction: Literal['minimize', 'maximize'] = 'maximize',
        sampler: Literal['grid', 'random', 'tpe'] = 'tpe',
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,  # sampler kwargs
    ):
        """
        Optimize experiment hyperparameters by processing data, generating explanations, evaluating with metrics, caching and retrieving the data.

        Args:
            data_ids (Union[int, Sequence[int]]): A single data ID or sequence of data IDs to specify the subset of data to process.
            explainer_id (int): An explainer ID to specify the explainer to use.
            metric_id (int): A metric ID to evaluate optimizer decisions.
            direction (Literal['minimize', 'maximize']): A string to specify the direction of optimization.
            sampler (Literal['grid', 'random', 'tpe']): A string to specify the sampler to use for optimization.
            n_trials (Optional[int]): An integer to specify the number of trials for optimization. If none passed, the number of trials is inferred from `timeout`.
            timeout (Optional[float]): A float to specify the timeout for optimization. Ignored, if `n_trials` is specified.

        Returns:
            The Experiment instance with updated results and state.

        This method orchestrates the experiment by configuring the manager, obtaining explainer and metric instances,
        processing data, generating explanations, and evaluating metrics. It then saves the results in the manager.

        Note: The input parameters allow for flexibility in specifying subsets of data, explainers, and metrics to process.
        If not provided, the method processes all available data, explainers, postprocessors, and metrics.
        """
        data_ids = [data_ids] if isinstance(data_ids, int) else data_ids
        data = self.manager.batch_data_by_ids(data_ids)
        explainer = self.manager.get_explainer_by_id(explainer_id)
        postprocessor = self.manager.get_postprocessor_by_id(
            0)  # sample postprocessor to ensure channel_dim
        metric = self.manager.get_metric_by_id(metric_id)

        objective = Objective(
            explainer=explainer,
            postprocessor=postprocessor,
            metric=metric,
            modality=self.modality,
            inputs=self.input_extractor(data),
            targets=self._get_targets(data_ids),
        )
        # TODO: grid search
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
            n_jobs=1,
        )
        opt_explainer = study.best_trial.user_attrs['explainer']
        opt_postprocessor = study.best_trial.user_attrs['postprocessor']
        return OptimizationOutput(
            explainer=opt_explainer,
            postprocessor=opt_postprocessor,
            study=study,
        )

    def get_inputs_flattened(self, data_ids: Optional[Sequence[int]] = None) -> Sequence[Tensor]:
        """
        Retrieve and flatten last run input data.

        Args:
            data_ids (Optional[Sequence[int]]): A sequence of data IDs to specify the subset of data to process.

        Returns:
            Flattened input data.

        This method retrieves input data using the input extractor and flattens it for further processing.

        Note: The input parameters allow for flexibility in specifying subsets of data to process. If not provided, the method processes all available data.
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

    def get_labels_flattened(self, data_ids: Optional[Sequence[int]] = None) -> Sequence[Tensor]:
        """
        Retrieve and flatten labels data.

        Args:
            data_ids (Optional[Sequence[int]]): A sequence of data IDs to specify the subset of data to process.

        Returns:
            Flattened labels data.

        This method retrieves label data using the label extractor and flattens it for further processing.
        """
        data, _ = self.manager.get_data(data_ids)
        labels = [self.label_extractor(datum) for datum in data]
        return self.manager.flatten_if_batched(labels, data)

    def get_targets_flattened(self, data_ids: Optional[Sequence[int]] = None) -> Sequence[Tensor]:
        """
        Retrieve and flatten target data.

        Args:
            data_ids (Optional[Sequence[int]]): A sequence of data IDs to specify the subset of data to process.

        Returns:
            Flattened target data.

        This method retrieves target data using the target extractor and flattens it for further processing.

        Note: The input parameters allow for flexibility in specifying subsets of data to process. If not provided, the method processes all available data.
        """
        if self.target_labels:
            return self.get_labels_flattened(data_ids)
        data, _ = self.manager.get_data(data_ids)
        targets = [self._get_targets(data_ids)]
        return self.manager.flatten_if_batched(targets, data)
        # targets = [self.label_extractor(datum) for datum in data] \
        #     if self.target_labels else [self._get_targets(data_ids)]
        # return self.manager.flatten_if_batched(targets, data)

    def get_outputs_flattened(self, data_ids: Optional[Sequence[int]] = None) -> Sequence[Tensor]:
        """
        Retrieve and flatten model outputs.

        Args:
            data_ids (Optional[Sequence[int]]): A sequence of data IDs to specify the subset of data to process.

        Returns:
            Flattened model outputs.

        This method retrieves flattened model outputs using the manager's get_flat_outputs method.

        Note: The input parameters allow for flexibility in specifying subsets of data to process. If not provided, the method processes all available data.
        """
        return self.manager.get_flat_outputs(data_ids)

    def get_explanations_flattened(self, data_ids: Optional[Sequence[int]] = None) -> Sequence[Sequence[Tensor]]:
        """
        Retrieve and flatten explanations from all explainers.

        Args:
            data_ids (Optional[Sequence[int]]): A sequence of data IDs to specify the subset of data to process.

        Returns:
            Flattened explanations from all explainers.

        This method retrieves flattened explanations for each explainer using the manager's `get_flat_explanations` method.

        Note: The input parameters allow for flexibility in specifying subsets of data to process. If not provided, the method processes all available data.
        """
        _, explainer_ids = self.manager.get_explainers()
        return [
            self.manager.get_flat_explanations(explainer_id, data_ids)
            for explainer_id in explainer_ids
        ]

    def get_evaluations_flattened(self, data_ids: Optional[Sequence[int]] = None) -> Sequence[Sequence[Sequence[Tensor]]]:
        """
        Retrieve and flatten evaluations for all explainers and metrics.

        Args:
            data_ids (Optional[Sequence[int]]): A sequence of data IDs to specify the subset of data to process.

        Returns:
            Flattened evaluations for all explainers and metrics.

        This method retrieves flattened evaluations for each explainer and metric using the manager's
        get_flat_evaluations method.

        Note: The input parameters allow for flexibility in specifying subsets of data to process. If not provided, the method processes all available data.
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

    @property
    def has_explanations(self):
        return self.manager.has_explanations
