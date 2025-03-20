from typing import Sequence, Optional, Callable, Any, List, Dict, Union
import torch
import inspect

from pnpxai.utils import format_into_tuple


def default_output_modifier(outputs):
    return outputs


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], torch.Tensor]] = None,
    ):
        super().__init__()
        forward_params = inspect.signature(model.forward).parameters

        self.model = model
        self.target_input_keys = target_input_keys or [next(iter(forward_params))]
        self.additional_input_keys = additional_input_keys or []
        self.output_modifier = output_modifier or default_output_modifier

        self._device = next(iter(model.parameters())).device
        self._required_order = self.target_input_keys + self.additional_input_keys
        self._validate_input_keys(forward_params)
        _key = next(iter(self._required_order))
        assert isinstance(_key, (str, int)), (
            f'Unsupported key type: {type(_key)}. Must be one of [str, int].',
        )
        self._key_type = type(_key)

    def _validate_input_keys(self, forward_params):
        for key in self._required_order:
            # validations for dict-like batch
            if isinstance(key, str):
                if key not in forward_params:
                    raise ValueError(
                        f"'input_key {key} not found in model forward params.")
    
            # validations for tuple-like batch
            elif isinstance(key, int):
                if key > len(forward_params)-1:
                    raise ValueError(
                        f"'input_key {key} must be lesser than {len(forward_params)-1}.")

    @property
    def device(self):
        return self._device

    @property
    def required_order(self):
        return self._required_order

    def extract_inputs(self, batch: Union[Sequence, Dict]):
        if isinstance(batch, Sequence):
            batch = list(batch)
            return tuple(batch[key].to(self.device) for key in self._required_order)
        else:
            return {key: batch[key].to(self.device) for key in self._required_order}

    def extract_target_inputs(self, batch: Union[Sequence, Dict]):
        if isinstance(batch, Sequence):
            batch = list(batch)
            return tuple(batch[key].to(self.device) for key in self.target_input_keys)
        else:
            return {key: batch[key].to(self.device) for key in self.target_input_keys}

    def extract_additional_inputs(self, batch: Union[Sequence, Dict]):
        if isinstance(batch, Sequence):
            batch = list(batch)
            return tuple(batch[key].to(self.device) for key in self.additional_input_keys)
        else:
            return {key: batch[key].to(self.device) for key in self.additional_input_keys}

    def format_inputs(self, inputs: Union[torch.Tensor, Sequence, Dict]):
        if isinstance(inputs, torch.Tensor):
            inputs = format_into_tuple(inputs)
        if isinstance(inputs, Sequence):
            inputs = list(inputs)
            return tuple(dict(zip(self._required_order, inputs)).values())
        return tuple(inputs[key].to(self.device) for key in self._required_order)

    def format_target_inputs(self, inputs: Union[torch.Tensor, Sequence, Dict]):
        if isinstance(inputs, torch.Tensor):
            inputs = format_into_tuple(inputs)
        if isinstance(inputs, Sequence):
            inputs = list(inputs)
            return tuple(dict(zip(self.target_input_keys, inputs)).values())
        return tuple(inputs[key].to(self.device) for key in self.target_input_keys)

    def format_additional_inputs(self, inputs: Union[Sequence, Dict]):
        if isinstance(inputs, torch.Tensor):
            inputs = format_into_tuple(inputs)
        if isinstance(inputs, Sequence):
            inputs = list(inputs)
        return tuple(
            inputs[key].to(self.device) for key in self.additional_input_keys)

    def forward(self, *formatted_inputs):
        if self._key_type is str:
            outputs = self.model.forward(
                **dict(zip(self._required_order, formatted_inputs)))
        elif self._key_type is int:
            outputs = self.model.forward(*formatted_inputs)
        return self.output_modifier(outputs)
