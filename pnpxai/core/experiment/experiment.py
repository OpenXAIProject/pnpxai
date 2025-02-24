from typing import Any, Callable, Optional, Sequence, Union, List, Literal, Type, Dict
import itertools
import inspect
from collections import defaultdict

import torch
from torch import Tensor
import numpy as np
import optuna

from pnpxai.core._types import DataSource, Model
from pnpxai.core.modality.modality import Modality
from pnpxai.core.experiment.experiment_metrics_defaults import (
    EVALUATION_METRIC_REVERSE_SORT,
    EVALUATION_METRIC_SORT_PRIORITY,
)
from pnpxai.core.experiment.manager import ExperimentManager
from pnpxai.core.utils import ModelWrapper
from pnpxai.explainers import Explainer, Lime, KernelShap
from pnpxai.explainers.base import Tunable
from pnpxai.explainers.utils import FunctionSelector
from pnpxai.explainers.utils.postprocess import Identity, PostProcessor
from pnpxai.explainers.types import (
    TensorOrTupleOfTensors,
    TargetLayerOrTupleOfTargetLayers,
)
from pnpxai.evaluator.optimizer.types import OptimizationOutput
from pnpxai.evaluator.optimizer.objectives import Objective
from pnpxai.evaluator.optimizer.utils import (
    load_sampler,
    get_default_n_trials,
)
from pnpxai.evaluator.metrics.base import Metric
from pnpxai.core.experiment.types import ExperimentOutput

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
        target_class_extractor (Optional[Callable[[Any], Any]]): Function to extract targets from data.
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
        explainers: Optional[Dict[str, Type[Explainer]]] = None,
        metrics: Optional[Dict[str, Type[Metric]]] = None,
        target_layer: Optional[TargetLayerOrTupleOfTargetLayers] = None,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], torch.Tensor]] = None,
        target_class_extractor: Optional[Callable[[Any], Any]] = None,
        label_key: Optional[Union[str, int]] = -1,
        target_labels: bool = False,
        cache_device: Optional[Union[torch.device, str]] = None,
    ):
        super(Experiment, self).__init__()

        # set model
        self.model = model
        self.model_device = next(self.model.parameters()).device
        self.target_input_keys = target_input_keys
        self.additional_input_keys = additional_input_keys
        self.output_modifier = output_modifier

        # set data
        self.manager = ExperimentManager(data=data, cache_device=cache_device)
        self.modality = modality

        # set explainer choices
        self.explainers = FunctionSelector()
        if explainers is not None:
            for k, explainer_type in explainers.items():
                self.explainers.add(k, explainer_type)

        # set metrics
        self.metrics = FunctionSelector()
        if metrics is not None:
            for k, metric_type in metrics.items():
                self.metrics.add(k, metric_type)

        self.target_layer = target_layer
        self.target_class_extractor = target_class_extractor or default_target_extractor
        self.target_labels = target_labels
        self.label_key = label_key

        self._explainer_key_to_id = {}
        self._postprocessor_key_to_id = {}
        self._metric_key_to_id = {}
        self.reset_errors()

    def reset_errors(self):
        self._errors: List[BaseException] = []

    @property
    def errors(self):
        return self._errors

    def to_device(self, x):
        return to_device(x, self.model_device)

    def _validate_choice(self, selector: FunctionSelector, choice: str):
        if choice not in selector.choices:
            raise ValueError(f"'{choice}' not found in {selector}")

    def run_batch(
        self,
        explainer_key: str,
        metric_key: str,
        data_ids: Optional[Sequence[int]] = None,
        pooling_method: Optional[str] = None,
        normalization_method: Optional[str] = None,
    ) -> ExperimentOutput:
        """
        Runs the experiment for selected batch of data, explainer_key, postprocessor and metric.

        Args:
            data_ids (Sequence[int]): A sequence of data IDs to specify the subset of data to process.
            explainer_key (str): ID of explainer to use for the run.
            postprocessor_id (int): ID of postprocessor to use for the run.
            metrics_id (int): ID of metric to use for the run.

        Returns:
            The dictionary of inputs, labels, outputs, targets, explainer, explanation, postprocessor, postprocessed, metric, and evaluation.

        This method orchestrates the experiment by configuring the manager, obtaining explainer and metric instances,
        processing data, generating explanations, and evaluating metrics. It then caches the results in the manager, and returns back to the user.

        Note: The input parameters allow for flexibility in specifying subset of data, explainer, postprocessor and metric to process.
        """

        # validate choices
        self._validate_choice(self.explainers, explainer_key)
        self._validate_choice(self.metrics, metric_key)

        data_ids = data_ids if data_ids is not None else self.manager.get_data_ids(
            data_ids)

        self.predict_batch(data_ids)
        _, explainer = self.explain_batch(
            data_ids, explainer_key, return_explainer=True)
        attrs_pp, pp_methods = self.postprocess_batch(
            data_ids, explainer_key, pooling_method, normalization_method,
            return_methods=True,
        )
        evals, metric = self.evaluate_batch(
            data_ids, explainer_key, metric_key,
            *pp_methods, return_metric=True,
        )
        return ExperimentOutput(
            explainer=explainer,
            metric=metric,
            explanations=attrs_pp,
            evaluations=evals,
        )

    @property
    def _wrapped_model(self):
        return ModelWrapper(
            model=self.model,
            target_input_keys=self.target_input_keys,
            additional_input_keys=self.additional_input_keys,
            output_modifier=self.output_modifier,
        )

    def input_extractor(self, batch) -> TensorOrTupleOfTensors:
        return self._wrapped_model.format_inputs(batch)

    def label_extractor(self, batch) -> Tensor:
        return batch[self.label_key]

    def _forward_batch(self, batch):
        formatted_inputs = self._wrapped_model.format_inputs(batch)
        return self._wrapped_model(*formatted_inputs)

    def predict_batch(
        self,
        data_ids: Sequence[int],
    ) -> TensorOrTupleOfTensors:
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
            batch = self.manager.batch_data_by_ids(data_ids_pred)
            outputs = self._forward_batch(batch)
            self.manager.cache_outputs(data_ids_pred, outputs)
        return self.manager.batch_outputs_by_ids(data_ids)

    def _create_instance(
            self,
            instance_type: Union[Type[Explainer], Type[Metric]],
            **kwargs,
    ) -> Union[Explainer, Metric]:
        # collect constructors from experiment
        required_params = dict(inspect.signature(instance_type).parameters)
        expr_params = dict(inspect.signature(self.__class__).parameters)
        params_base = {}
        for param_nm, param in expr_params.items():
            if param_nm in required_params:
                required_param = required_params.pop(param_nm)
                params_base[required_param.name] = getattr(self, required_param.name)

        # update additional constructors
        additional_params = defaultdict(list)
        modalities = format_into_tuple(self.modality)
        for param_nm, param in required_params.items():
            if param_nm in kwargs:
                if param_nm in modalities[0].util_functions:
                    assert isinstance(kwargs[param_nm], Sequence), (
                        f"'{param_nm}' must be a tuple."
                    )
                    assert len(kwargs[param_nm]) == len(modalities), (
                        f"'{param_nm}' must have same length with modality."
                    )
                additional_params[param_nm] = kwargs[param_nm]
            else:
                if param_nm in modalities[0].util_functions:
                    for modality in modalities:
                        default_fn_key = modality.util_functions[param_nm].choices[0]
                        default_fn = modality.util_functions[param_nm].select(
                            default_fn_key,
                        )
                        additional_params[param_nm].append(default_fn)
                elif param.default is inspect.Parameter.empty:
                    # raise TypeError when returning
                    continue
                else:
                    additional_params[param_nm] = param.default

        # format additional constructors
        additional_params = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in additional_params.items()
        }
        return instance_type(**params_base, **additional_params)

    def create_explainer(self, explainer_key: str, **kwargs) -> Explainer:
        explainer_type = self.explainers.get(explainer_key)
        explainer = self._create_instance(explainer_type, **kwargs)

        # set util function space for tunable explainer
        modalities = format_into_tuple(self.modality)
        if isinstance(explainer, Tunable):
            for tunable_param in explainer.tunable_params:
                if not tunable_param.is_leaf and tunable_param.space is None:
                    # tunable_param may be one of tunable util functions
                    fn_key, *_keys = tunable_param.name.split('.')
                    if _keys:
                        modality_key, *_ = _keys
                    else:
                        modality_key = 0
                    modality = modalities[int(modality_key)]
                    tunable_param.set_selector(
                        modality.util_functions[fn_key], set_space=True)
        return explainer

    def create_metric(self, metric_key, **kwargs) -> Metric:
        metric_type = self.metrics.get(metric_key)
        metric = self._create_instance(metric_type, **kwargs)

        # modify pooling dim
        modalities = format_into_tuple(self.modality)
        pooling_dims = tuple()
        if hasattr(metric, 'pooling_dim'):
            for modality in modalities:
                pooling_dims += (modality.pooling_dim,)
        metric.pooling_dim = format_out_tuple_if_single(pooling_dims)
        return metric

    def explain_batch(
        self,
        data_ids: Sequence[int],
        explainer_key: str,
        return_explainer: bool = True,
        **kwargs,
    ) -> TensorOrTupleOfTensors:
        """
        Explains selected batch of data within experiment.

        Args:
            data_ids (Sequence[int]): A sequence of data IDs to specify the subset of data to process.

        Returns:
            Batched model explanations corresponding to data ids.

        This method orchestrates the experiment by configuring the manager, obtaining explainer instance,
        processing data, and generating explanations. It then caches the results in the manager, and returns back to the user.
        """

        # collect data_ids not cached yet
        explainer_id = self._explainer_key_to_id.get(explainer_key, '_placeholder')
        data_ids_expl = [
            idx for idx in data_ids
            if self.manager.get_explanation_by_id(idx, explainer_id) is None
        ]
        if len(data_ids_expl) > 0:
            batch = self.manager.batch_data_by_ids(data_ids_expl)
            inputs = self._wrapped_model.extract_inputs(batch)
            targets = self.get_targets_by_id(data_ids_expl)
            explainer = self.create_explainer(explainer_key, **kwargs)

            if explainer_id == '_placeholder':
                explainer_id = self.manager.add_explainer(explainer)
                self._explainer_key_to_id[explainer_key] = explainer_id

            attrs = explainer.attribute(inputs, targets)
            self.manager.cache_explanations(
                explainer_id, data_ids_expl, attrs)
        attrs = self.manager.batch_explanations_by_ids(data_ids, explainer_id)
        if return_explainer:
            explainer = self.manager.get_explainer_by_id(explainer_id)
            return attrs, explainer
        return attrs

    def postprocess_batch(
        self,
        data_ids: List[int],
        explainer_key: str,
        pooling_method: Optional[str] = None,
        normalization_method: Optional[str] = None,
        return_methods: bool = False,
    ) -> TensorOrTupleOfTensors:
        """
        Postprocesses selected batch of data within experiment.

        Args:
            data_ids (Sequence[int]): A sequence of data IDs to specify the subset of data to postprocess.
            explainer_key (str): An explainer ID to specify the explainer to use.
            postprocessor_id (int): A postprocessor ID to specify the postprocessor to use.
            return_methods (bool): Returns postprocess methods if True

        Returns:
            Batched postprocessed model explanations corresponding to data ids.

        This method orchestrates the experiment by configuring the manager, obtaining explainer instance,
        processing data, and generating explanations. It then caches the results in the manager, and returns back to the user.
        """
        explainer_id = self._explainer_key_to_id.get(explainer_key)
        pp_key = self._generate_postprocessor_key(
            pooling_method, normalization_method)
        pp_id = self._postprocessor_key_to_id.get(pp_key, '_placeholder')
        mods = format_into_tuple(self.modality)
        pools = format_into_tuple(pooling_method)
        norms = format_into_tuple(normalization_method)
        attrs = self.manager.batch_explanations_by_ids(
            data_ids, explainer_id)
        attrs = format_into_tuple(attrs)
        zipped = itertools.zip_longest(mods, pools, norms, attrs)
        attrs_pp = tuple()
        applied_pps = tuple()
        applied_pools = tuple()
        applied_norms = tuple()
        explainer_type = self.explainers.get(explainer_key)
        for modality, pool, norm, attr in zipped:
            # given or default pooling method
            pool = pool or modality.util_functions['pooling_fn'].choices[0]

            # TODO: more elegant way
            skip_pool = (
                explainer_type in (KernelShap, Lime)
                and modality.dtype_key == int
            )
            if skip_pool:
                pool = 'identity'
                modality.util_functions['pooling_fn'].add_fallback_option(
                    key=pool, value=Identity)
            norm = norm or modality.util_functions['normalization_fn'].choices[0]
            pp = PostProcessor(modality, pool, norm)
            attrs_pp += (pp(attr),)
            applied_pps += (pp,)
            applied_pools += (pool,)
            applied_norms += (norm,)
        attrs_pp = format_out_tuple_if_single(attrs_pp)
        if pp_id == '_placeholder':
            pp_id = self.manager.add_postprocessor(applied_pps)
            pp_key = self._generate_postprocessor_key(applied_pools, applied_norms)
            self._postprocessor_key_to_id[pp_key] = pp_id
        if not return_methods:
            return attrs_pp
        applied_pools = format_out_tuple_if_single(applied_pools)
        applied_norms = format_out_tuple_if_single(applied_norms)
        return attrs_pp, (applied_pools, applied_norms)

    def _generate_postprocessor_key(self, pooling_method, normalization_method):
        pools = format_into_tuple(pooling_method)
        norms = format_into_tuple(normalization_method)
        return '-'.join(
            ['_'.join([pool, norm]) for pool, norm in zip(pools, norms)])

    def evaluate_batch(
        self,
        data_ids: List[int],
        explainer_key: str,
        metric_key: str,
        pooling_method: Optional[str] = None,
        normalization_method: Optional[str] = None,
        return_metric: bool = False,
    ) -> TensorOrTupleOfTensors:
        """
        Evaluates selected batch of data within experiment.

        Args:
            data_ids (Sequence[int]): A sequence of data IDs to specify the subset of data to postprocess.
            explainer (str): An explainer ID to specify the explainer to use.
            postprocessor_id (int): A postprocessor ID to specify the postprocessor to use.
            metric (int): A metric ID to evaluate the model explanations.

        Returns:
            Batched model evaluations corresponding to data ids.

        This method orchestrates the experiment by configuring the manager, obtaining explainer instance,
        processing data, generating explanations, and evaluating results. It then caches the results in the manager, and returns back to the user.
        """
        explainer_id = self._explainer_key_to_id.get(explainer_key)
        pp_id =self._postprocessor_key_to_id.get(self._generate_postprocessor_key(
            pooling_method, normalization_method,
        ))
        metric_id = self._metric_key_to_id.get(metric_key, '_placeholder')
        data_ids_eval = [
            idx for idx in data_ids
            if self.manager.get_evaluation_by_id(
                idx, explainer_id, pp_id, metric_id) is None
        ]
        if len(data_ids_eval) > 0:
            batch = self.manager.batch_data_by_ids(data_ids_eval)
            inputs = self._wrapped_model.extract_inputs(batch)
            targets = self.get_targets_by_id(data_ids_eval)
            attrs_pp = self.postprocess_batch(
                data_ids_eval, explainer_key, pooling_method, normalization_method)
            explainer = self.manager.get_explainer_by_id(explainer_id)
            metric = self.create_metric(metric_key)
            if metric_id == '_placeholder':
                metric_id = self.manager.add_metric(metric)
                self._metric_key_to_id[metric_key] = metric_id
            evals = metric.set_explainer(explainer).evaluate(
                inputs, targets, attrs_pp,
            )
            self.manager.cache_evaluations(
                explainer_id, pp_id, metric_id,
                data_ids_eval, evals,
            )
        evals = self.manager.batch_evaluations_by_ids(
            data_ids, explainer_id, pp_id, metric_id
        )
        if return_metric:
            metric = self.manager.get_metric_by_id(metric_id)
            return evals, metric
        return evals

    def get_targets_by_id(self, data_ids):
        if self.target_labels:
            batch = self.manager.batch_data_by_ids(data_ids)
            labels = batch[self.label_key]
            return labels.to(self.model_device)
        outputs = self.manager.batch_outputs_by_ids(data_ids)
        return self.target_class_extractor(outputs).to(self.model_device)

    def is_tunable(self, explainer_key):
        is_tunable_explainer = issubclass(
            self.explainers.get(explainer_key), Tunable)
        modalities = format_into_tuple(self.modality)
        is_tunable_by_pool = any(
            len(modality.util_functions['pooling_fn'].choices) > 1
            for modality in modalities
        )
        is_tunable_by_norm = any(
            len(modality.util_functions['normalization_fn'].choices) > 1
            for modality in modalities
        )
        return any([
            is_tunable_explainer,
            is_tunable_by_pool,
            is_tunable_by_norm,
        ])

    def optimize(
        self,
        explainer_key: str,
        metric_key: str,
        direction: Literal['minimize', 'maximize'] = 'maximize',
        sampler: Literal['grid', 'random', 'tpe'] = 'tpe',
        data_ids: Optional[Union[int, Sequence[int]]] = None,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        num_threads: Optional[int] = None,
        show_progress: bool = False,
        **kwargs,  # sampler kwargs
    ) -> OptimizationOutput:
        """
        Optimize experiment hyperparameters by processing data, generating explanations, evaluating with metrics, caching and retrieving the data.

        Args:
            data_ids (Union[int, Sequence[int]]): A single data ID or sequence of data IDs to specify the subset of data to process.
            explainer (str): An explainer ID to specify the explainer to use.
            metric (int): A metric ID to evaluate optimizer decisions.
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
        org_num_threads = torch.get_num_threads()
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        objective = Objective(
            modality=self.modality,
            explainer=self.create_explainer(explainer_key),
            metric=self.create_metric(metric_key),
            data=self.manager.get_data(data_ids)[0],
            target_class_extractor=self.target_class_extractor,
            label_key=self.label_key,
            target_labels=self.target_labels,
            show_progress=show_progress,
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
        torch.set_num_threads(org_num_threads)
        return OptimizationOutput(
            explainer=opt_explainer,
            postprocessor=opt_postprocessor,
            study=study,
        )

    def get_inputs_flattened(
        self,
        data_ids: Optional[Sequence[int]] = None,
    ) -> Sequence[Tensor]:
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
        data = [format_out_tuple_if_single(
            self.input_extractor(datum)) for datum in data]
        return self.manager.flatten_if_batched(data, data)

    def get_labels_flattened(
        self,
        data_ids: Optional[Sequence[int]] = None,
    ) -> Sequence[Tensor]:
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

    def get_targets_flattened(
        self,
        data_ids: Optional[Sequence[int]] = None,
    ) -> Sequence[Tensor]:
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
        targets = [self.get_targets_by_id(data_ids)]
        return self.manager.flatten_if_batched(targets, data)

    def get_outputs_flattened(
        self,
        data_ids: Optional[Sequence[int]] = None,
    ) -> Sequence[Tensor]:
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

    def get_explanations_flattened(
        self,
        data_ids: Optional[Sequence[int]] = None,
    ) -> Sequence[Sequence[Tensor]]:
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
            self.manager.get_flat_explanations(explainer, explainer_ids)
            for explainer in explainer_ids
        ]

    def get_evaluations_flattened(
        self,
        data_ids: Optional[Sequence[int]] = None,
    ) -> Sequence[Sequence[Sequence[Tensor]]]:
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
                explainer, postprocessor_id, metric, data_ids)
            for metric in metric_ids
        ] for postprocessor_id in postprocessor_ids]
            for explainer in explainer_ids]

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
