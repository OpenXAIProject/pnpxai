from typing import Optional, List

import torch
from torch import Tensor, eye, ones
from torch.nn import Module
from zennit.attribution import Attributor, Gradient
from zennit.canonizers import Canonizer, SequentialMergeBatchNorm

from .base import ExplainerInterface
from ._utils.lrp.composites import SUPPORTED_COMPOSITES
from ._utils.lrp.attributors import SUPPORTED_ATTRIBUTORS


class LRPBase(ExplainerInterface):
    def __init__(
        self,
        model: Module,
        attributor: Attributor,
        composite_type: Optional[str],
        canonizers: Optional[List[Canonizer]],
    ):
        super().__init__(
            model = model,
        )
        self._set_attributor(attributor)
        self.composite_type = composite_type
        self._set_canonizers(canonizers)
    
    def attribute(
        self,
        inputs: Tensor,
        targets: Tensor,
        **tunables
    ) -> Tensor:
        output_shape = self._get_output_shape(inputs.shape[1:])
        composite = self._get_composite(**tunables)
        with self.attributor(model=self.model, composite=composite) as attributor:
            _, attributions = attributor(inputs, eye(output_shape)[[targets]])
        return attributions
    
    def _get_output_shape(self, input_shape) -> int:
        sample_inputs = torch.ones(1, *input_shape)
        outputs = self.model(sample_inputs)
        return outputs.shape[-1]
    
    def _get_composite(self, **tunables):
        composite_cls = SUPPORTED_COMPOSITES.get(self.composite_type)
        if composite_cls:
            return composite_cls(**tunables, canonizers=self.canonizers)
        return
    
    def _set_attributor(self, attributor):
        self.attributor = attributor if attributor else Gradient
    
    def _set_canonizers(self, canonizers):
        self.canonizers = canonizers if canonizers else [SequentialMergeBatchNorm()]


class LRPZero(LRPBase):
    TUNABLES = None
    def __init__(
        self,
        model: Module,
        attributor: Optional[Attributor] = None,
        canonizers: Optional[List[Canonizer]] = None,
    ):
        super().__init__(
            model = model,
            attributor = attributor,
            composite_type = None,
            canonizers = canonizers
        )


class LRPEpsilon(LRPBase):
    TUNABLES = {'epsilon'}
    
    def __init__(
        self,
        model: Module,
        attributor: Optional[Attributor] = None,
        canonizers: Optional[List[Canonizer]] = None,
    ):
        super().__init__(
            model = model,
            attributor = attributor,
            composite_type = 'uniform_epsilon',
            canonizers = canonizers
        )


class LRPGamma(LRPBase):
    TUNABLES = {'gamma'}
    
    def __init__(
        self,
        model: Module,
        attributor: Optional[Attributor] = None,
        canonizers: Optional[List[Canonizer]] = None,
    ):
        super().__init__(
            model = model,
            attributor = attributor,
            composite_type = 'uniform_gamma',
            canonizers = canonizers
        )


class LRPEpsilonGammaBox(LRPBase):
    TUNABLES = {'low', 'high', 'epsilon', 'gamma'}
    
    def __init__(
        self,
        model: Module,
        attributor: Optional[Attributor] = None,
        canonizers: Optional[List[Canonizer]] = None,
    ):
        super().__init__(
            model = model,
            attributor = attributor,
            composite_type = 'epsilon_gamma_box',
            canonizers = canonizers
        )

    def attribute(
        self,
        inputs: Tensor,
        targets: Tensor,
        **tunables
    ) -> Tensor:
        output_shape = self._get_output_shape(inputs.shape[1:])
        if not tunables:
            tunables = dict(high=3., low=-3.)
        composite = self._get_composite(**tunables)
        with self.attributor(model=self.model, composite=composite) as attributor:
            _, attributions = attributor(inputs, eye(output_shape)[[targets]])
        return attributions


class LRPEpsilonPlus(LRPBase):
    TUNABLES = {'epsilon'}
    
    def __init__(
        self,
        model: Module,
        attributor: Optional[Attributor] = None,
        canonizers: Optional[List[Canonizer]] = None,
    ):
        super().__init__(
            model = model,
            attributor = attributor,
            composite_type = 'epsilon_plus',
            canonizers = canonizers
        )


class LRPEpsilonAlpha2Beta1(LRPBase):
    TUNABLES = {'epsilon'}
    
    def __init__(
        self,
        model: Module,
        attributor: Optional[Attributor] = None,
        canonizers: Optional[List[Canonizer]] = None,
    ):
        super().__init__(
            model = model,
            attributor = attributor,
            composite_type = 'epsilon_alpha2_beta1',
            canonizers = canonizers
        )


class LRPEpsilonAlpha2Beta1Flat(LRPBase):
    TUNABLES = {'epsilon'}
    
    def __init__(
        self,
        model: Module,
        attributor: Optional[Attributor] = None,
        canonizers: Optional[List[Canonizer]] = None,
    ):
        super().__init__(
            model = model,
            attributor = attributor,
            composite_type = 'epsilon_alpha2_beta1_flat',
            canonizers = canonizers
        )


# [TO-DO] name map composite for custom lrp



SUPPORTED_LRP = {
    'lrp_zero': LRPZero,
    'lrp_epsilon': LRPEpsilon,
    'lrp_gamma': LRPGamma,
    'lrp_epsilon_gamma_box': LRPEpsilonGammaBox,
    'lrp_epsilon_plus': LRPEpsilonPlus,
    'lrp_epsilon_alpha2_beta1': LRPEpsilonAlpha2Beta1,
    'lrp_epsilon_alph2_beta1_flat': LRPEpsilonAlpha2Beta1Flat,
}


