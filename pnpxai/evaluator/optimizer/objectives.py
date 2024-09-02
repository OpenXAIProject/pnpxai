from typing import Optional, Tuple, Dict, Callable, Any
from torch import Tensor
from optuna.trial import Trial, TrialState

from pnpxai.core._types import TensorOrTupleOfTensors
from pnpxai.core.modality.modality import Modality
from pnpxai.explainers.base import Explainer
from pnpxai.explainers import KernelShap, Lime
from pnpxai.explainers.utils.postprocess import PostProcessor
from pnpxai.evaluator.metrics.base import Metric
from pnpxai.evaluator.optimizer.utils import generate_param_key, nest_params
from pnpxai.evaluator.optimizer.suggestor import suggest
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single


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

    def __call__(self, trial: Trial) -> float:
        explainer = suggest(trial, self.explainer, self.modality, key=self.EXPLAINER_KEY)
        postprocessor = format_out_tuple_if_single(tuple(suggest(
                trial, postprocessor, modality,
                key=generate_param_key(
                    self.POSTPROCESSOR_KEY,
                    modality.__class__.__name__ if len(format_into_tuple(self.postprocessor)) > 1 else None
                )
            ) for postprocessor, modality in zip(
                format_into_tuple(self.postprocessor),
                format_into_tuple(self.modality),
        )))

        # ignore duplicated samples
        states_to_consider = (TrialState.COMPLETE,)
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                trial.set_user_attr('explainer', explainer)
                trial.set_user_attr('postprocessor', postprocessor)
                return t.value

        # explain and postprocess
        attrs = format_into_tuple(
            explainer.attribute(self.inputs, self.targets)
        )
        pass_pooling = isinstance(explainer, (KernelShap, Lime))
        postprocessed = tuple(
            pp.normalization_fn(attr) if pass_pooling else pp(attr)
            for pp, attr in zip(
                format_into_tuple(postprocessor),
                format_into_tuple(attrs),
            )
        )

        if any(pp.isnan().sum() > 0 or pp.isinf().sum() > 0 for pp in postprocessed):
            '''
            Treat a failure as nan.
            Failure might occur when postprocessed result containing non-countable value
            such as nan or inf.
            '''
            return float('nan')

        postprocessed = format_out_tuple_if_single(postprocessed)
        metric = self.metric.set_explainer(explainer)
        evals = format_into_tuple(
            metric.evaluate(self.inputs, self.targets, postprocessed)
        )

        # keep current explainer and postprocessor on trial
        trial.set_user_attr('explainer', explainer)
        trial.set_user_attr('postprocessor', postprocessor)
        return (sum(*evals) / len(evals)).item()

