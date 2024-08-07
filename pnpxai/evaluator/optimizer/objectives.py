from typing import Optional, Tuple, Dict, Callable, Any
from torch import Tensor
from optuna.trial import Trial

from pnpxai.core._types import TensorOrTupleOfTensors
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils.postprocess import PostProcessor
from pnpxai.evaluator.metrics.base import Metric


class Objective:
    def __init__(
        self,
        explainer: Explainer,
        metric: Metric,
        suggest_methods: Dict[str, Callable],
        channel_dim: int=1,
        inputs: Optional[TensorOrTupleOfTensors]=None,
        targets: Optional[Tensor]=None,
    ):
        self.explainer = explainer
        self.metric = metric
        self.suggest_methods = suggest_methods
        self.channel_dim = channel_dim
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

    def suggest(self, trial: Trial) -> Tuple[Dict[str, Any]]:
        explainer_kwargs = {
            k: suggest(trial) for k, suggest in self.suggest_methods.items()
        }
        postprocessor_kwargs = {
            'pooling_method': explainer_kwargs.pop('pooling_method', 'sumpos'),
            'normalization_method': explainer_kwargs.pop('normalization_method', 'minmax'),
            'channel_dim': self.channel_dim,
        }
        return explainer_kwargs, postprocessor_kwargs

    def __call__(self, trial: Trial) -> float:
        explainer_kwargs, postprocessor_kwargs = self.suggest(trial)
        explainer = self.explainer.set_kwargs(**explainer_kwargs)
        metric = self.metric.set_explainer(explainer)
        postprocessor = PostProcessor(**postprocessor_kwargs)
        attrs = explainer.attribute(self.inputs, self.targets)
        postprocessed = postprocessor(attrs)
        evals = metric.evaluate(self.inputs, self.targets, postprocessed)
        return evals.sum().item()
