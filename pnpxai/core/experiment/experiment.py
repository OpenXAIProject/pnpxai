from typing import Any, Callable, Optional, Sequence, Union, List
import time
import warnings
import itertools

import torch
from torch import Tensor
import numpy as np
from plotly import express as px
from plotly.graph_objects import Figure

from pnpxai.core.detector.types import SklearnModel, SklearnRegressor, SklearnClassifier
from pnpxai.core.experiment.experiment_metrics_defaults import EVALUATION_METRIC_REVERSE_SORT, EVALUATION_METRIC_SORT_PRIORITY
from pnpxai.core.experiment.observable import ExperimentObservableEvent
from pnpxai.utils import class_to_string, Observable, to_device, format_into_tuple
from pnpxai.messages import get_message
from pnpxai.core.experiment.manager import ExperimentManager
from pnpxai.explainers.base import Explainer
from pnpxai.metrics.base import Metric
from pnpxai.core._types import DataSource, Model, ModalityOrListOfModalities
from pnpxai.explainers.utils.postprocess import postprocess_attr
from pnpxai.explainers.sklearn.utils import format_into_array


def default_input_extractor(x):
    return x[0]


def default_label_extractor(x):
    return x[-1]


def default_target_extractor(y):
    return y.argmax(-1)


def format_into_array_if_sklearn(func):
    def wrapper(self, *args, **kwargs):
        out = func(self, *args, **kwargs)
        if self.is_sklearn_model:
            return format_into_array(out)
        return out
    return wrapper


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
        self.model_device = None if self.is_sklearn_model \
            else next(self.model.parameters()).device

        self.manager = ExperimentManager(
            data=data,
            explainers=explainers,
            metrics=metrics or [],
            cache_device=cache_device,
        )

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
    def is_sklearn_model(self):
        return isinstance(self.model, SklearnModel)

    @property
    def errors(self):
        return self._errors

    @property
    def all_explainers(self) -> Sequence[Explainer]:
        return self.manager.all_explainers

    @property
    def all_metrics(self) -> Sequence[Metric]:
        return self.manager.all_metrics

    def get_current_explainers(self) -> List[Explainer]:
        return self.manager.get_explainers()[0]

    def get_current_metrics(self) -> List[Metric]:
        return self.manager.get_metrics()[0]

    def to_device(self, x):
        if self.is_sklearn_model:
            return x
        return to_device(x, self.model_device)

    def run(
        self,
        data_ids: Optional[Sequence[int]] = None,
        explainer_ids: Optional[Sequence[int]] = None,
        metrics_ids: Optional[Sequence[int]] = None,
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
        self.manager.set_config(data_ids, explainer_ids, metrics_ids)
        
        # inference
        data, data_ids = self.manager.get_data_to_predict()
        outputs = self._predict(data)
        self.manager.save_outputs(outputs, data, data_ids)

        # explain
        explainers, explainer_ids = self.manager.get_explainers()

        for explainer, explainer_id in zip(explainers, explainer_ids):
            explainer_name = class_to_string(explainer)
            data, data_ids = self.manager.get_data_to_process_for_explainer(
                explainer_id)
            explanations = self._explain(data, data_ids, explainer)
            self.manager.save_explanations(
                explanations, data, data_ids, explainer_id
            )
            message = get_message(
                'experiment.event.explainer', explainer=explainer_name
            )
            print(f"[Experiment] {message}")
            self.fire(ExperimentObservableEvent(
                self.manager, message, explainer))

            metrics, metric_ids = self.manager.get_metrics()
            for metric, metric_id in zip(metrics, metric_ids):
                metric_name = class_to_string(metric)
                data, data_ids = self.manager.get_data_to_process_for_metric(
                    explainer_id, metric_id)
                explanations, data_ids = self.manager.get_valid_explanations(
                    explainer_id, data_ids)
                data, _ = self.manager.get_data(data_ids)
                evaluations = self._evaluate(
                    data, data_ids, explanations, explainer, metric)
                self.manager.save_evaluations(
                    evaluations, data, data_ids, explainer_id, metric_id)

                message = get_message(
                    'experiment.event.explainer.metric', explainer=explainer_name, metric=metric_name)
                print(f"[Experiment] {message}")
                self.fire(ExperimentObservableEvent(
                    self.manager, message, explainer, metric))
        return self

    @property
    def _predict_fn(self):
        if isinstance(self.model, SklearnClassifier):
            return self.model.predict_proba
        elif isinstance(self.model, SklearnRegressor):
            return self.model.predict
        else:
            return self.model

    def _predict(self, data: DataSource):
        outputs = [
            self._predict_fn(*format_into_tuple(
                self.to_device(self.input_extractor(datum)))
            ) for datum in data
        ]
        return outputs

    def _explain(self, data: DataSource, data_ids: List[int], explainer: Explainer):
        explanations = [None] * len(data)
        explainer_name = class_to_string(explainer)
        for i, (datum, data_id) in enumerate(zip(data, data_ids)):
            datum = self.to_device(datum)
            inputs = self.input_extractor(datum)
            targets = self.label_extractor(datum) if self.target_labels \
                else self._get_targets(data_ids)
            explanation = explainer.attribute(
                inputs=inputs,
                targets=targets,
            )
            if self.is_sklearn_model:
                explanation = format_into_array(explanation)
            explanations[i] = explanation
            
            # try:
            #     datum = self.to_device(datum)
            #     inputs = self.input_extractor(datum)
            #     targets = self.label_extractor(datum) if self.target_labels \
            #         else self._get_targets(data_ids)
            #     explanation = explainer.attribute(
            #         inputs=inputs,
            #         targets=targets,
            #     )
            #     if self.is_sklearn_model:
            #         explanation = format_into_array(explanation)
            #     explanations[i] = explanation
            # except NotImplementedError as error:
            #     warnings.warn(
            #         f"\n[Experiment] {get_message('experiment.errors.explainer_unsupported', explainer=explainer_name)}")
            #     raise error
            # except Exception as e:
            #     warnings.warn(
            #         f"\n[Experiment] {get_message('experiment.errors.explanation', explainer=explainer_name, error=e)}")
            #     self._errors.append(e)
        return explanations

    def _evaluate(self, data: DataSource, data_ids: List[int], explanations: DataSource, explainer: Explainer, metric: Metric):
        if explanations is None:
            return None
        started_at = time.time()
        metric_name = class_to_string(metric)
        explainer_name = class_to_string(explainer)

        evaluations = [None] * len(data)
        for i, (datum, data_id, explanation) in enumerate(zip(data, data_ids, explanations)):
            if explanation is None:
                continue
            datum = self.to_device(datum)
            explanation = self.to_device(explanation)
            inputs = self.input_extractor(datum)
            targets = self.label_extractor(datum) if self.target_labels \
                else self._get_targets(data_ids)
            
            def explain_func(model, inputs, targets):
                if isinstance(inputs, np.ndarray):
                    inputs = torch.tensor(inputs)
                if isinstance(targets, np.ndarray):
                    targets = torch.tensor(targets)

                attr = explainer.attribute(
                    inputs=inputs,
                    targets=targets
                )
                if isinstance(attr, Tensor):
                    attr = attr.detach().cpu().numpy()

                return attr


            
            try:
                # [GH] input args as kwargs to compute metric in an experiment
                evaluations[i] = metric.evaluate(
                    inputs=inputs,
                    targets=targets,
                    attributions=explanation,
                    explain_func=explain_func,
                )
            except Exception as e:
                warnings.warn(
                    f"\n[Experiment] {get_message('experiment.errors.evaluation', explainer=explainer_name, metric=metric_name, error=e)}")
                self._errors.append(e)
        elapsed_time = time.time() - started_at
        print(
            f"[Experiment] {get_message('elapsed', modality=metric_name, elapsed=elapsed_time)}")

        return evaluations

    @property
    def _stack_data(self):
        if self.is_sklearn_model:
            return format_into_array
        return torch.stack

    def _get_targets(self, data_ids: List[int]):
        outputs = self._stack_data([
            self.manager._cache.get_output(data_id)
            for data_id in data_ids
        ])
        return self.target_extractor(outputs)

    def get_visualizations_flattened(self) -> Sequence[Sequence[Figure]]:
        """
        Generate flattened visualizations for each data point based on explanations.

        Returns:
            List of visualizations for each data point across explainers.

        This method retrieves valid explanations for each explainer, formats them for visualization,
        and generates visualizations using Plotly Express. The results are flattened based on the data points
        and returned as a list of lists of figures.
        """
        assert self.modality == 'image', f"Visualization for '{self.modality} is not supported yet"
        explainers, explainer_ids = self.manager.get_explainers()
        # Get all data ids
        experiment_data_ids = self.manager.get_data_ids()
        visualizations = []

        for explainer, explainer_id in zip(explainers, explainer_ids):
            # Get all valid explanations and data ids for this explainer
            explanations, data_ids = self.manager.get_valid_explanations(
                explainer_id)
            data = self.manager.get_data(data_ids)[0]
            explainer_visualizations = []
            for explanation in explanations:
                figs = []
                for attr in explanation:
                    postprocessed = postprocess_attr(
                        attr,
                        channel_dim=0,
                        pooling_method='l2normsq',
                        normalization_method='minmax'
                    ).detach().numpy()
                    fig = px.imshow(postprocessed, color_continuous_scale='RdBu_R', color_continuous_midpoint=.5)
                    figs.append(fig)
                explainer_visualizations.append(figs)

            # explainer_visualizations = [[
            #     px.imshow(
            #         postprocess_attr(
            #             attr, # C x H x W
            #             channel_dim=0,
            #             pooling_method='l2normsq',
            #             normalization_method='minmax',
            #         ).detach().numpy(),
            #         color_continuous_scale='RdBu_R',
            #         color_continuous_midpoint=.5,
            #     ) for attr in explanation]
            #     for explanation in explanations
            # ]
            # # Visualize each valid explanataion
            # for datum, explanation in zip(data, explanations):
            #     inputs = self.input_extractor(datum)
            #     targets = self.target_extractor(datum)
            #     formatted = explainer.format_outputs_for_visualization(
            #         inputs=inputs,
            #         targets=targets,
            #         explanations=explanation,
            #         modality=self.modality
            #     )

            #     if not self.manager.is_batched:
            #         formatted = [formatted]
            #     formatted_visualizations = [
            #         px.imshow(explanation, color_continuous_scale="RdBu_r", color_continuous_midpoint=0.0) for explanation in formatted
            #     ]
            #     if not self.manager.is_batched:
            #         formatted_visualizations = formatted_visualizations[0]
            #     explainer_visualizations.append(formatted_visualizations)

            flat_explainer_visualizations = self.manager.flatten_if_batched(
                explainer_visualizations, data)
            # Set visualizaions of all data ids as None
            explainer_visualizations = {
                idx: None for idx in experiment_data_ids}
            # Fill all valid visualizations
            for visualization, data_id in zip(flat_explainer_visualizations, data_ids):
                explainer_visualizations[data_id] = visualization
            visualizations.append(list(explainer_visualizations.values()))

        return visualizations

    def get_inputs_flattened(self) -> Sequence[Tensor]:
        """
        Retrieve and flatten last run input data.

        Returns:
            Flattened input data.

        This method retrieves input data using the input extractor and flattens it for further processing.
        """
        data, _ = self.manager.get_data()
        data = [self.input_extractor(datum) for datum in data]
        if self.is_sklearn_model:
            data = [format_into_array(d) for d in data]
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
        if self.is_sklearn_model:
            data = [format_into_array(d) for d in data]
        return self.manager.flatten_if_batched(data, data)

    def get_labels_flattened(self) -> Sequence[Tensor]:
        data, data_ids = self.manager.get_data()
        labels = [self.label_extractor(datum) for datum in data]
        if self.is_sklearn_model:
            labels = [format_into_array(label) for label in labels]
        return self.manager.flatten_if_batched(labels, data)

    def get_targets_flattened(self) -> Sequence[Tensor]:
        """
        Retrieve and flatten target data.

        Returns:
            Flattened target data.

        This method retrieves target data using the target extractor and flattens it for further processing.
        """
        if self.target_labels:
            return self.get_labels_flattened()
        data, data_ids = self.manager.get_data()
        targets = [self._get_targets(data_ids)]
        if self.is_sklearn_model:
            targets = [format_into_array(target) for target in targets]
        return self.manager.flatten_if_batched(targets, data)
        # targets = [self.label_extractor(datum) for datum in data] \
        #     if self.target_labels else [self._get_targets(data_ids)]
        # return self.manager.flatten_if_batched(targets, data)

    def get_outputs_flattened(self) -> Sequence[Tensor]:
        """
        Retrieve and flatten model outputs.

        Returns:
            Flattened model outputs.

        This method retrieves flattened model outputs using the manager's get_flat_outputs method.
        """
        return self.manager.get_flat_outputs()

    def get_explanations_flattened(self) -> Sequence[Sequence[Tensor]]:
        """
        Retrieve and flatten explanations from all explainers.

        Returns:
            Flattened explanations from all explainers.

        This method retrieves flattened explanations for each explainer using the manager's get_flat_explanations method.
        """
        _, explainer_ids = self.manager.get_explainers()
        return [
            self.manager.get_flat_explanations(explainer_id)
            for explainer_id in explainer_ids
        ]

    def get_evaluations_flattened(self) -> Sequence[Sequence[Sequence[Tensor]]]:
        """
        Retrieve and flatten evaluations for all explainers and metrics.

        Returns:
            Flattened evaluations for all explainers and metrics.

        This method retrieves flattened evaluations for each explainer and metric using the manager's
        get_flat_evaluations method.
        """
        _, explainer_ids = self.manager.get_explainers()
        _, metric_ids = self.manager.get_metrics()

        formatted = [[
            self.manager.get_flat_evaluations(explainer_id, metric_id)
            for metric_id in metric_ids
        ] for explainer_id in explainer_ids]

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
    def is_image_task(self):
        return self.modality == 'image'

    @property
    def has_explanations(self):
        return self.manager.has_explanations

    @property
    def records(self):
        return ExperimentRecords(self)


class ExperimentRecords:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        # self._zipped_data = zip(
        #     self.experiment.manager.get_data()[-1], # data_id; data_ids[data_loc]
        #     self.experiment.get_inputs_flattened(), # input; inputs[data_loc]
        #     self.experiment.get_labels_flattened(), # label; labels[data_loc]
        #     self.experiment.get_outputs_flattened(), # output; outputs[data_loc]
        #     self.experiment.get_targets_flattened(), # target; targets[data_loc]
        #     self._rearrange_explanations(),
        #     self._rearrange_evaluations(),
        # )

    @property
    def _metric_exists(self):
        return len(self.experiment.all_metrics) > 0

    @property
    def _base_data(self):
        return [
            self.experiment.manager.get_data()[-1], # data_id; data_ids[data_loc]
            self.experiment.get_inputs_flattened(), # input; inputs[data_loc]
            self.experiment.get_labels_flattened(), # label; labels[data_loc]
            self.experiment.get_outputs_flattened(), # output; outputs[data_loc]
            self.experiment.get_targets_flattened(), # target; targets[data_loc]
            self._rearrange_explanations(),
        ]

    @property
    def _zipped_data(self):
        if not self._metric_exists:
            return zip(*self._base_data)
        data = self._base_data + [self._rearrange_evaluations()]
        return zip(*data)

    def _rearrange_explanations(self):
        # expls[explainer_loc][data_loc] -> expls[data_loc][explainer_loc]
        rearranged = []
        for expl_loc, expl_by_data in enumerate(self.experiment.get_explanations_flattened()):
            for data_loc, expl in enumerate(expl_by_data):
                if len(rearranged) < data_loc + 1:
                    rearranged.append([])
                if len(rearranged[data_loc]) < expl_loc + 1:
                    rearranged[data_loc].append([])
                rearranged[data_loc][expl_loc].append(expl)
        return rearranged

    def _rearrange_evaluations(self):
        # evals[explainer_loc][metric_loc][data_loc] -> evals[data_loc][explainer_loc][metric_loc]
        rearranged = []
        for expl_loc, eval_by_expl in enumerate(self.experiment.get_evaluations_flattened()):
            for metric_loc, eval_by_data in enumerate(eval_by_expl):
                for data_loc, metric in enumerate(eval_by_data):
                    if len(rearranged) < data_loc + 1:
                        rearranged.append([])
                    if len(rearranged[data_loc]) < expl_loc + 1:
                        rearranged[data_loc].append([])
                    if len(rearranged[data_loc][expl_loc]) < metric_loc + 1:
                        rearranged[data_loc][expl_loc].append([])
                    rearranged[data_loc][expl_loc][metric_loc].append(metric)
        return rearranged

    @property
    def explainers(self):
        return {
            explainer_id: class_to_string(explainer)
            for explainer, explainer_id
            in zip(*self.experiment.manager.get_explainers())
        }

    @property
    def metrics(self):
        return {
            metric_id: class_to_string(metric)
            for metric, metric_id
            in zip(*self.experiment.manager.get_metrics())
        }

    def __len__(self):
        return len(self.experiment.manager.get_data()[-1])

    def __getitem__(self, data_loc):
        row = next(itertools.islice(self._zipped_data, data_loc, None))
        zipped_explanations = zip(self.explainers.items(), row[5])
        explanations = []
        data = {
            'data_id': row[0],
            'input': row[1],
            'label': row[2],
            'output': row[3],
            'target': row[4],
            'explanations': explanations,
        }
        for explainer_loc, ((explainer_id, explainer_nm), explanation) in enumerate(zipped_explanations):
            explanation = {
                'explainer_id': explainer_id,
                'explainer_nm': explainer_nm,
                'value': explanation[0]
            }
            if self._metric_exists:
                zipped_evaluations = zip(self.metrics.items(), row[6][explainer_loc])
                evaluations = []
                for (metric_id, metric_nm), evaluation in zipped_evaluations:
                    evaluations.append({
                        'metric_id': metric_id,
                        'metric_nm': metric_nm,
                        'value': evaluation[0],
                    })
                explanation['evaluations'] = evaluations
            explanations.append(explanation)
        return data



            
        # return {
        #     'data_id': row[0],
        #     'input': row[1],
        #     'label': row[2],
        #     'output': row[3],
        #     'target': row[4],
        #     'explanations': [{
        #         'explainer_id': explainer_id,
        #         'explainer_nm': explainer_nm,
        #         'value': explanation[0],
        #         'evaluations': [{
        #             'metric_id': metric_id,
        #             'metric_nm': metric_nm,
        #             'value': evaluation[0],
        #         } for (metric_id, metric_nm), evaluation in zipped_evaluations(explainer_loc)]
        #     } for explainer_loc, ((explainer_id, explainer_nm), explanation) in enumerate(zipped_explanations)]
        # }

