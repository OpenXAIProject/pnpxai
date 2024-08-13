from typing import Callable, Tuple, Union, Optional, Dict, Any

import torch
from torch import Tensor
from torch.nn.modules import Module
from captum.attr import KernelShap as CaptumKernelShap
from optuna.trial import Trial

from pnpxai.explainers.types import ForwardArgumentExtractor
from pnpxai.explainers.utils.baselines import BaselineMethodOrFunction
from pnpxai.explainers.utils.feature_masks import FeatureMaskMethodOrFunction
from pnpxai.evaluator.optimizer.utils import generate_param_key
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single
from .base import Explainer

class KernelShap(Explainer):
    def __init__(
        self,
        model: Module,
        n_samples: int=25,
        baseline_fn: Union[BaselineMethodOrFunction, Tuple[BaselineMethodOrFunction]]='zeros',
        feature_mask_fn: Union[FeatureMaskMethodOrFunction, Tuple[FeatureMaskMethodOrFunction]]='felzenszwalb',
        forward_arg_extractor: Optional[ForwardArgumentExtractor]=None,
        additional_forward_arg_extractor: Optional[ForwardArgumentExtractor]=None,
        mask_token_id: Optional[int]=None,
    ) -> None:
        super().__init__(
            model,
            forward_arg_extractor,
            additional_forward_arg_extractor
        )
        self.n_samples = n_samples
        self.baseline_fn = baseline_fn
        self.feature_mask_fn = feature_mask_fn
        self.mask_token_id = mask_token_id

    def attribute(
        self,
        inputs: Tensor,
        targets: Optional[Tensor]=None,
    ) -> Union[Tensor, Tuple[Tensor]]:
        forward_args, additional_forward_args = self._extract_forward_args(inputs)
        forward_args = format_into_tuple(forward_args)
        explainer = CaptumKernelShap(self.model)
        attrs = explainer.attribute(
            inputs=forward_args,
            target=targets,
            baselines=self._get_baselines(forward_args),
            feature_mask=self._get_feature_masks(forward_args),
            n_samples=self.n_samples,
            additional_forward_args=additional_forward_args,
        )
        attrs = format_out_tuple_if_single(attrs)
        return attrs

    def suggest_tunables(self, trial: Trial, key: Optional[str]=None) -> Dict[str, Any]:
        baseline_fns = format_into_tuple(self._load_baseline_fn())
        feature_mask_fns = format_into_tuple(self._load_feature_mask_fn())
        return {
            'n_samples': trial.suggest_int(
                generate_param_key(key, 'n_samples'),
                low=10, high=50, step=10,
            ),
            'baseline_fn': format_out_tuple_if_single(tuple(
                baseline_fn.suggest_tunables(
                    trial=trial,
                    key=generate_param_key(key, 'baseline_fn', order),
                ) for order, baseline_fn in enumerate(baseline_fns)
            )),
            'feature_mask_fn': format_out_tuple_if_single(tuple(
                feature_mask_fn.suggest_tunables(
                    trial=trial,
                    key=generate_param_key(key, 'feature_mask_fn', order),
                ) for order, feature_mask_fn in enumerate(feature_mask_fns)
            )),
        }