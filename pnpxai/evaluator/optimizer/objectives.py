from typing import Optional, Tuple, Dict, Callable, Any
from torch import Tensor
from optuna.trial import Trial

from pnpxai.core._types import TensorOrTupleOfTensors
from pnpxai.core.modality.modality import Modality
from pnpxai.explainers.base import Explainer
from pnpxai.explainers import KernelShap, Lime
from pnpxai.explainers.utils.postprocess import PostProcessor
from pnpxai.evaluator.metrics.base import Metric
from pnpxai.evaluator.optimizer.utils import generate_param_key, nest_params
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single
from pnpxai.evaluator.optimizer.suggestor import suggest_explainer_params


class Objective:
    EXPLAINER_KEY = 'explainer'
    POSTPROCESSOR_KEY = 'postprocessor'

    def __init__(
        self,
        explainer: Explainer,
        postprocessor: PostProcessor,
        metric: Metric,
        modality: Modality,
        inputs: Optional[TensorOrTupleOfTensors] = None,
        targets: Optional[Tensor] = None,
    ):
        self.explainer = explainer
        self.postprocessor = postprocessor
        self.metric = metric
        self.modality = modality
        self.inputs = inputs
        self.targets = targets

    def set_inputs(self, inputs):
        self.inputs = inputs
        return self

    def set_targets(self, targets):
        self.targets = targets
        return self

    def set_data(self, inputs, targets):
        self.set_inputs(inputs)
        self.set_targets(targets)
        return self

    def format_explainer_kwargs(self, kwargs):
        formatted = kwargs.copy()
        for k, v in kwargs.items():
            if k == 'baseline_fn':
                formatted[k] = self._format_baseline_fn(v)
            if k == 'feature_mask_fn':
                formatted[k] = self._format_feature_mask_fn(v)
        return formatted

    def _format_baseline_fn(self, kwargs):
        return format_out_tuple_if_single(tuple(
            baseline_fn.set_kwargs(**_kwargs)
            for baseline_fn, _kwargs
            in zip(
                format_into_tuple(self.explainer.baseline_fn),
                format_into_tuple(kwargs),
            )))

    def _format_feature_mask_fn(self, kwargs):
        return format_out_tuple_if_single(tuple(
            feature_mask_fn.set_kwargs(**_kwargs)
            for feature_mask_fn, _kwargs
            in zip(
                format_into_tuple(self.explainer.feature_mask_fn),
                format_into_tuple(kwargs),
            )))

    def format_postprocessor_kwargs(self, kwargs):
        return kwargs

    def load_from_optuna_params(self, optuna_params):
        nested = nest_params(optuna_params.copy())
        if nested.get(self.EXPLAINER_KEY):
            explainer_kwargs = {
                k: format_out_tuple_if_single(tuple(list(v.values())))
                if k in ['baseline_fn', 'feature_mask_fn']
                else v for k, v in nested[self.EXPLAINER_KEY].items()
            }
        else:
            explainer_kwargs = {}
        explainer = self.explainer.set_kwargs(
            **self.format_explainer_kwargs(explainer_kwargs)
        )
        postprocessor = format_out_tuple_if_single(tuple(
            pp.set_kwargs(
                **self.format_postprocessor_kwargs(pp_kwargs))
            for pp, pp_kwargs in zip(
                format_into_tuple(self.postprocessor),
                nested[self.POSTPROCESSOR_KEY].values(),
            )
        ))
        return explainer, postprocessor

    def __call__(self, trial: Trial) -> float:
        explainer_kwargs = suggest_explainer_params(
            self.explainer, self.modality, trial, key=self.EXPLAINER_KEY
        )
        postprocessor_kwargs = tuple(
            self.modality.suggest_tunable_post_processors(
                trial=trial, key=generate_param_key(self.POSTPROCESSOR_KEY, order)
            )
            for order, pp in enumerate(format_into_tuple(self.postprocessor))
        )
        explainer, postprocessor = self.load_from_optuna_params(trial.params)

        # explain and postprocess
        attrs = format_into_tuple(
            explainer.attribute(self.inputs, self.targets)
        )
        pass_pooling = isinstance(explainer, (KernelShap, Lime))

        postprocessed = format_out_tuple_if_single(tuple(
            pp.normalize_attributions(attr) if pass_pooling else pp(attr)
            for pp, attr in zip(
                format_into_tuple(postprocessor),
                format_into_tuple(attrs),
            )
        ))

        metric = self.metric.set_explainer(explainer)
        evals = format_into_tuple(
            metric.evaluate(self.inputs, self.targets, postprocessed)
        )
        return (sum(*evals) / len(evals)).item()
