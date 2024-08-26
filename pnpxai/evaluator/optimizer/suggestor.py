from typing import Optional
from optuna import Trial

from pnpxai.evaluator.optimizer.utils import generate_param_key
from pnpxai.core.modality.modality import Modality
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils.baselines import BaselineFunction
from pnpxai.explainers.utils.feature_masks import FeatureMaskFunction
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single


def _suggest_explainer_feature_mask(name: str, explainer: Explainer, modality: Modality, trial: Trial, key: Optional[str] = None):
    feature_mask_fns = format_into_tuple(explainer.load_feature_mask_fn())
    return format_out_tuple_if_single(tuple(
        modality.suggest_tunable_feature_masks(
            trial=trial,
            key=generate_param_key(key, name, order),
        )
        for order, feature_mask_fn in enumerate(feature_mask_fns)
    ))


def _suggest_explainer_baseline(name: str, explainer: Explainer, modality: Modality, trial: Trial, key: Optional[str] = None):
    baseline_fns = format_into_tuple(explainer.load_baseline_fn())
    return format_out_tuple_if_single(tuple(
        modality.suggest_tunable_baselines(
            trial=trial,
            key=generate_param_key(key, name, order),
        )
        for order, baseline_fn in enumerate(baseline_fns)
    ))


def suggest_explainer_params(explainer: Explainer, modality: Modality, trial: Trial, key: Optional[str] = None):
    params = {}
    for name, method_data in explainer.TUNABLES:
        method_type, method_kwargs = method_data
        if method_type == BaselineFunction:
            params[name] = _suggest_explainer_baseline(
                name=name, explainer=explainer, modality=modality, trial=trial, key=key
            )
            continue

        if method_type == FeatureMaskFunction:
            params[name] = _suggest_explainer_feature_mask(
                name=name, explainer=explainer, modality=modality, trial=trial, key=key
            )
            continue

        method = {
            list: trial.suggest_categorical,
            int: trial.suggest_int,
            float: trial.suggest_float,
        }.get(method_type, None)

        if method is None and callable(method_type):
            params[name] = method_type(
                trial=trial, key=generate_param_key(key, name), **method_kwargs
            )
            continue

        params[name] = method(
            name=generate_param_key(key, name), **method_kwargs
        )

    return params
