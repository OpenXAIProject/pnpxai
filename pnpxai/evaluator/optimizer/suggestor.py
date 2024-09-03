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
from pnpxai.explainers import Lime, KernelShap
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
    key: Optional[str] = None,
    force_params: Optional[Dict[str, Any]] = None,
):
    """
    A utility function that suggests parameters for a given object based on an optimization trial. 
    The function recursively tunes the parameters of the object according to the modality (or 
    modalities) provided.

    Parameters:
        trial (Trial):
            The trial object from an optimization framework like Optuna, used to suggest 
            parameters for tuning.
        obj (Any):
            The object whose parameters are being tuned. This object must implement 
            `get_tunables()` and `set_kwargs()` methods.
        modality (Union[Modality, Tuple[Modality]]):
            The modality (e.g., image, text) or tuple of modalities the object is operating on. 
            If multiple modalities are provided, the function handles multi-modal tuning.
        key (Optional[str], optional):
            An optional key to uniquely identify the set of parameters being tuned, 
            useful for differentiating parameters in multi-modal scenarios. Defaults to None.

    Returns:
        Any:
            The object with its parameters set according to the trial suggestions.
    
    Notes:
        - The function uses `map_suggest_method` to map the tuning method based on the method 
          type provided in the tunables.
        - It supports multi-modal tuning, where different modalities may require different 
          parameters to be tuned. 
        - For utility functions (`UtilFunction`), the function further tunes parameters 
          based on the selected function from the modality.
    
    Example:
        Assuming `trial` is an instance of `optuna.trial.Trial`, and `explainer: Explainer` is an object 
        with tunable parameters, you can tune it as follows:

        ```python
        tuned_explainer = suggest(trial, explainer, modality)
        ```
    """

    is_multi_modal = len(format_into_tuple(modality)) > 1
    force_params = force_params or {}
    for param_nm, (method_type, method_kwargs) in obj.get_tunables().items():
        if param_nm in force_params:
            param = force_params[param_nm]
        else:
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
                    )  # update param_nm
                    _method = map_suggest_method(trial, _method_type)
                    fn_nm = _method(
                        name=generate_param_key(key, _param_nm),
                        **_method_kwargs
                    )
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
