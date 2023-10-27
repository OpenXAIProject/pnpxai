from typing import Optional, List
from zennit.core import Hook
from zennit.canonizers import Canonizer
from zennit.composites import (
    LayerMapComposite,
    NameMapComposite,
    EpsilonGammaBox,
    EpsilonPlus,
    EpsilonAlpha2Beta1,
    EpsilonAlpha2Beta1Flat,
)
from zennit.composites import layer_map_base
from zennit.types import Convolution, Linear, MaxPool
from zennit.rules import Gamma, Epsilon


class UniformComposite(LayerMapComposite):
    def __init__(
        self,
        rule: Hook,
        canonizers: Optional[List[Canonizer]] = None
    ):
        self.rule = rule
        super().__init__(
            layer_map = self._get_layer_map(),
            canonizers = canonizers
        )

    def _get_layer_map(self):
        additional_layer_map = [
        	(layer_type, self.rule)
        	for layer_type in [Convolution, Linear, MaxPool]
        ]
        return layer_map_base() + additional_layer_map


class UniformEpsilon(UniformComposite):
    def __init__(
        self,
        epsilon: float = .25,
        zero_params = None,
        canonizers: Optional[List[Canonizer]] = None,
    ):
        self.epsilon = epsilon
        self.zero_params = zero_params
        super().__init__(
            rule = Epsilon(epsilon=epsilon, zero_params=zero_params),
            canonizers = canonizers,
        )


class UniformGamma(UniformComposite):
    def __init__(
        self,
        gamma: float = .25,
        stabilizer: float = 1e-6,
        zero_params = None,
        canonizers: Optional[List[Canonizer]] = None,
    ):
        self.gamma = gamma
        self.stabilizer = stabilizer
        self.zero_params = zero_params
        super().__init__(
            rule = Gamma(gamma=gamma, stabilizer=stabilizer, zero_params=zero_params),
            canonizers = canonizers,
        )
        

SUPPORTED_COMPOSITES = {
	'uniform_epsilon': UniformEpsilon,
	'uniform_gamma': UniformGamma,
	'epsilon_gamma_box': EpsilonGammaBox,
	'epsilon_plus': EpsilonPlus,
	'epsilon_alpha2_beta1': EpsilonAlpha2Beta1,
	'epsilon_alpha2_beta1_flat': EpsilonAlpha2Beta1Flat,
	'name_map': NameMapComposite,
}
 
