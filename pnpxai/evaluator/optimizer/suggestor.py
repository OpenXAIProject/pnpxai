from typing import Optional, Any, Dict, Type, Union, Tuple
from optuna import Trial

from pnpxai.evaluator.optimizer.utils import generate_param_key
from pnpxai.core.modality.modality import Modality
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils import (
    UtilFunction,
    BaselineFunction,
    FeatureMaskFunction,
    PoolingFunction,
    NormalizationFunction,
)
from pnpxai.explainers.utils.function_selectors import FunctionSelector
from pnpxai.evaluator.optimizer.utils import generate_param_key
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single


def map_suggest_method(
    trial: Trial,
    method_type: Type[Any],
):
    return {
        list: trial.suggest_categorical,
        int: trial.suggest_int,
        float: trial.suggest_float,
    }.get(method_type, None)


def map_fn_selector(
    modality: Modality,
    method_type: Type[Any],
):
    return {
        BaselineFunction: modality.baseline_fn_selector,
        FeatureMaskFunction: modality.feature_mask_fn_selector,
        PoolingFunction: modality.pooling_fn_selector,
        NormalizationFunction: modality.normalization_fn_selector,
    }.get(method_type, None)


def suggest(
    trial: Trial,
    obj: Any,
    modality: Union[Modality, Tuple[Modality]],
    key: Optional[str]=None,
):
    is_multi_modal = len(format_into_tuple(modality)) > 1
    for param_nm, (method_type, method_kwargs) in obj.get_tunables().items():
        method = map_suggest_method(trial, method_type)
        if method is not None:
            param = method(
                name=generate_param_key(key, param_nm),
                **method_kwargs
            )
        elif issubclass(method_type, UtilFunction):
            param = []
            for mod in format_into_tuple(modality):
                fn_selector = map_fn_selector(mod, method_type)
                _param_nm, (_method_type, _method_kwargs) = next(
                    iter(fn_selector.get_tunables().items())
                )
                _param_nm = generate_param_key(
                    param_nm,
                    mod.__class__.__name__ if is_multi_modal else None,
                    _param_nm,
                ) # update param_nm
                _method = map_suggest_method(trial, _method_type)
                fn_nm = _method(
                    name=generate_param_key(key, _param_nm),
                    **_method_kwargs
                )

                # # TODO: safe way to assign non-varying fn_kwargs
                # fn_kwargs = {}
                # if fn_nm == 'mean':
                #     fn_kwargs['dim'] = mod.channel_dim
                # if fn_nm == 'token':
                #     fn_kwargs['token_id'] = mod.mask_token_id
                # if param_nm == 'pooling_fn':
                #     fn_kwargs['channel_dim'] = mod.channel_dim
                fn = fn_selector.select(fn_nm)
                param.append(suggest(
                    trial, fn, mod,
                    key=generate_param_key(
                        key, param_nm,
                        mod.__class__.__name__ if is_multi_modal else None,
                    ),
                ))
            param = format_out_tuple_if_single(tuple(param))
        obj = obj.set_kwargs(**{param_nm: param})
    return obj