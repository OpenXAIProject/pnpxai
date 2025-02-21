from typing import Tuple, Optional, Union, Type, Dict, List, Callable, Any
import abc
from abc import abstractmethod

import sys
import inspect
from collections import defaultdict

from torch import Tensor
from torch.nn.modules import Module
import optuna

from pnpxai.core.utils import ModelWrapper
from pnpxai.explainers.types import TunableParameter
from pnpxai.utils import generate_param_key


# Ensure compatibility with Python 2/3
ABC = abc.ABC if sys.version_info >= (3, 4) else abc.ABCMeta(str("ABC"), (), {})


NON_DISPLAYED_ATTRS = [
    "model",
    "forward_arg_extractor",
    "additional_forward_arg_extractor",
    "device",
    "n_classes",
    "zennit_composite",
    "_wrapped_model",
    "_tunable_params",
]


class Explainer(ABC):
    """
    Abstract base class for implementing attribution explanations for machine learning models.

    This class provides methods for extracting forward arguments, loading baseline and feature mask functions, and applying them during attribution.

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract forward arguments from inputs.
        additional_forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract additional forward arguments.
        **kwargs: Additional keyword arguments to pass to the constructor.

    Notes:
        - Subclasses must implement the `attribute` method to define how attributions are computed.
        - The `forward_arg_extractor` and `additional_forward_arg_extractor` functions allow for customization in extracting forward arguments from the inputs.
    """
    SUPPORTED_MODULES: List[Type[Module]] = []
    SUPPORTED_DTYPES: List[Type] = []
    SUPPORTED_NDIMS: List[int] = []

    def __init__(
        self,
        model: Module,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
        **kwargs,
    ) -> None:
        self.model = model.eval()
        self.target_input_keys = target_input_keys
        self.additional_input_keys = additional_input_keys
        self.output_modifier = output_modifier

    @property
    def _wrapped_model(self):
        return ModelWrapper(
            model=self.model,
            target_input_keys=self.target_input_keys,
            additional_input_keys=self.additional_input_keys,
            output_modifier=self.output_modifier,
        )

    @property
    def wrapped_model(self):
        return self._wrapped_model

    @property
    def device(self):
        return next(self.model.parameters()).device

    def __repr__(self):
        kwargs_repr = ', '.join(
            '{}={}'.format(key, value)
            for key, value in self.__dict__.items()
            if key not in NON_DISPLAYED_ATTRS and value is not None
        )
        return "{}({})".format(self.__class__.__name__, kwargs_repr)

    def format_inputs(self, inputs: Union[Tuple, Dict]):
        forward_args = self._wrapped_model.format_target_inputs(inputs)
        additional_forward_args = self._wrapped_model.format_additional_inputs(inputs)
        return forward_args, additional_forward_args

    @abstractmethod
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor]],
        targets: Tensor,
    ) -> Union[Tensor, Tuple[Tensor]]:
        """
        Computes attributions for the given inputs and targets.

        Args:
            inputs (Union[Tensor, Tuple[Tensor]]): The inputs for the model.
            targets (Tensor): The target labels.

        Returns:
            Union[Tensor, Tuple[Tensor]]: The computed attributions.
        """
        raise NotImplementedError

    def is_tunable(self):
        return isinstance(self, Tunable)


class Tunable:
    def __init__(self, params: Optional[List[TunableParameter]] = None):
        self._tunable_params = params or []

    @property
    def tunable_params(self):
        return [tp for tp in self._tunable_params if not tp.disabled]

    @property
    def non_tunable_params(self):
        tunable_params = [tp.name.split('.')[0] for tp in self.tunable_params]
        required_params = [
            param for param in inspect.signature(self.__class__).parameters]
        return [param for param in required_params if param not in tunable_params]

    def get_current_tunable_param_values(self):
        tps = {}
        for tp in self._tunable_params:
            if tp.is_leaf:
                tps[tp.name] = tp.current_value
                continue
            if not hasattr(tp, 'selector'):
                tps[tp.name] = tp.current_value
                continue
            tps[tp.name] = tp.selector.get_key(tp.current_value.__class__)
            if isinstance(tp.current_value, Tunable):
                leaf_values = tp.current_value.get_current_tunable_param_values()
                for k, v in leaf_values.items():
                    tps[f'{tp.name}.{k}'] = v
        return tps

    def register_tunable_params(
        self,
        params: List[Union[TunableParameter, Tuple[TunableParameter]]],
    ):
        for param in params:
            self._register_tunable_param(param)

    def _register_tunable_param(
        self,
        param: Union[TunableParameter, Tuple[TunableParameter]],
        key=None,
    ):
        if isinstance(param, tuple):
            for i, _param in enumerate(param):
                self._register_tunable_param(_param, i)
        else:
            if key is not None:
                param.rename(f'{param.name}.{key}')
            self._tunable_params.append(param)

    def disable_tunable_param(self, param_name):
        for tp in self._tunable_params:
            if tp.name.startswith(param_name):
                tp.disable()

    def enable_tunable_param(self, param_name):
        for tp in self._tunable_params:
            if tp.name.startswith(param_name):
                tp.enable()

    def suggest(self, trial: optuna.Trial, key=None):
        suggested = defaultdict(tuple)
        for tp in self.tunable_params:
            suggest = {
                str: trial.suggest_categorical,
                int: trial.suggest_int,
                float: trial.suggest_float
            }.get(tp.dtype)
            suggested_value = suggest(
                name=generate_param_key(key, tp.name),
                **tp.space
            )
            if not tp.is_leaf:  # maybe util functions
                # init default util function
                suggested_value = tp.selector.select(suggested_value)
                # recursively suggest
                if isinstance(suggested_value, Tunable):
                    suggested_value = suggested_value.suggest(trial, key=tp.name)
            nm, *keys = tp.name.split('.')
            if keys:
                suggested[nm] += (suggested_value,)
            else:
                suggested[nm] = suggested_value

        non_tunable_params = {k: getattr(self, k) for k in self.non_tunable_params}
        return self.__class__(
            **non_tunable_params,
            **suggested,
        )
