from typing import Dict, Any

from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils.postprocess import RELEVANCE_POOLING_METHODS
from pnpxai.explainers.utils.baselines import DEFAULT_BASELINE_METHODS

DEFAULT_SUGGEST_FUNCTION_CONFIGS = {
    'n_iter': {
        'method': 'suggest_int',
        'low': 10,
        'high': 100,
        'step': 10,
    },
    'n_samples': {
        'method': 'suggest_int',
        'low': 10,
        'high': 100,
        'step': 10,
    },
    'n_steps': {
        'method': 'suggest_int',
        'low': 10,
        'high': 100,
        'step': 10,
    },
    'noise_level': {
        'method': 'suggest_float',
        'low': .1,
        'high': 1.,
        'step': .1,
    },
    'epsilon': {
        'method': 'suggest_float',
        'low': 1e-9,
        'high': 1.,
        'log': True,
    },
    'gamma': {
        'method': 'suggest_float',
        'low': 1e-9,
        'high': 1.,
        'log': True,
    },
    'baseline_fn': {
        'method': 'suggest_categorical',
        'choices': list(DEFAULT_BASELINE_METHODS.keys()),
    },
    'pooling_method': {
        'method': 'suggest_categorical',
        'choices': list(RELEVANCE_POOLING_METHODS.keys()),
    },
}

DEFAULT_LOGSCALE_BASE = 10

def _get_suggest_method(param_nm, func_conf):
    def suggest(trial):
        _func_conf = func_conf.copy()
        method = _func_conf.pop('method')
        return getattr(trial, method)(param_nm, **_func_conf)
    return suggest


def create_default_suggest_methods(explainer: Explainer):
    suggest_methods = {}
    for param_nm, func_conf in DEFAULT_SUGGEST_FUNCTION_CONFIGS.items():
        if hasattr(explainer, param_nm) or param_nm == 'pooling_method':
            suggest_methods[param_nm] = _get_suggest_method(param_nm, func_conf)
    return suggest_methods

def create_default_search_space(explainer: Explainer):
    search_space = {}
    for param_nm, func_conf in DEFAULT_SUGGEST_FUNCTION_CONFIGS.items():
        if hasattr(explainer, param_nm) or param_nm == 'pooling_method':
            _func_conf = func_conf.copy()
            method_nm = _func_conf.pop('method')
            if method_nm is 'suggest_int':
                search_space[param_nm] = list(range(
                    _func_conf['low'],
                    _func_conf['high']+_func_conf['step'],
                    _func_conf['step'],
                ))
            elif method_nm is 'suggest_float':
                ls = [_func_conf['low']]
                while ls[-1] < _func_conf['high']:
                    if _func_conf.get('log', False):
                        ls.append(ls[-1]*DEFAULT_LOGSCALE_BASE)
                    else:
                        ls.append(ls[-1]+_func_conf['step'])
                search_space[param_nm] = ls
            elif method_nm is 'suggest_categorical':
                search_space[param_nm] = _func_conf['choices']
    return search_space
    
