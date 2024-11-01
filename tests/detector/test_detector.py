from torch import nn
from typing import Sequence, Any

from pnpxai.core.detector import detect_model_architecture, types

from transformers.models.bert.modeling_bert import BertAttention, BertConfig
from transformers.models.visual_bert.modeling_visual_bert import VisualBertAttention, VisualBertConfig
from transformers.models.vilt.modeling_vilt import ViltAttention, ViltConfig


class TestModelArchitectureDetector:
    def _get_final_subclasses(self, source):
        if not hasattr(source, "__subclass__"):
            return [source]

        subclasses = []
        for subclass in source.__subclass__:
            subclasses.extend(self._get_final_subclasses(subclass))

        return subclasses

    def _init_module(self, module: nn.Module, args: Sequence[Any]):
        if issubclass(module, BertAttention):
            return module(BertConfig())
        if issubclass(module, VisualBertAttention):
            return module(VisualBertConfig())
        if issubclass(module, ViltAttention):
            return module(ViltConfig())

        return module(*args)

    def test_detector_module_types(self):
        configs = (
            (types.Linear, (1, 1)),
            (types.Convolution, (1, 1, 1)),
            (types.LSTM, (1, 1)),
            (types.RNN, (1, 1)),
            (types.Attention, (1, 1)),
        )

        for d_type, type_args in configs:
            modules = self._get_final_subclasses(d_type)
            print(d_type, modules)
            for module in modules:
                model = self._init_module(module, type_args)
                detected = detect_model_architecture(model)
                print(module, detected, flush=True)
                assert d_type in detected
