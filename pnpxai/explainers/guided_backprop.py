from typing import Optional

from torch.nn.modules import Module
from zennit.core import BasicHook, Stabilizer
from zennit.composites import LayerMapComposite
from zennit.rules import NoMod
from zennit.types import Linear

from .lrp import LRPBase, canonizers_base

from .types import (
    ForwardArgumentExtractor,
    TargetLayer,
)


class GuidedBackpropRule(BasicHook):
    def __init__(self, stabilizer=1e-6, zero_params=None) -> None:
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod(zero_params=zero_params)],
            output_modifiers=[lambda output: output.clamp(min=0)],
            gradient_mapper=(lambda out_grad, outputs: out_grad.clamp(min=0) / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )


class GuidedBackprop(LRPBase):
    def __init__(
        self,
        model: Module,
        stabilizer: float=1e-6,
        forward_arg_extractor: Optional[ForwardArgumentExtractor]=None,
        additional_forward_arg_extractor: Optional[ForwardArgumentExtractor]=None,
        layer: Optional[TargetLayer]=None,
        n_classes: Optional[int]=None,
    ) -> None:
        self.stabilizer = stabilizer
        super().__init__(
            model,
            self._composite,
            forward_arg_extractor,
            additional_forward_arg_extractor,
            layer,
            n_classes
        )

    @property
    def _layer_map(self):
        return [(Linear, GuidedBackpropRule(stabilizer=self.stabilizer))]

    @property
    def _canonizers(self):
        return canonizers_base()

    @property
    def _composite(self):
        return LayerMapComposite(
            layer_map=self._layer_map,
            canonizers=self._canonizers,
        )