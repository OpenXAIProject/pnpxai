from abc import abstractmethod
from typing import Literal, Dict, Any, Union, Callable

import optuna

from pnpxai.utils import format_into_tuple, format_out_tuple_if_single


AVAILABLE_SAMPLERS = {
    'grid': optuna.samplers.GridSampler,
    'random': optuna.samplers.RandomSampler,
    'tpe': optuna.samplers.TPESampler,
}

DEFAULT_N_TRIALS = {
    'grid': None,
    'random': 50,
    'tpe': 50,
}

def load_sampler(sampler: Literal['grid', 'random', 'tpe']='tpe', **kwargs):
    return AVAILABLE_SAMPLERS[sampler](**kwargs)

def get_default_n_trials(sampler):
    return DEFAULT_N_TRIALS[sampler]

def nest_params(flattened_params):
    nested = {}
    for k, v in flattened_params.items():
        ref = nested
        splits = k.split('.')
        while splits:
            s = splits.pop(0)
            if s not in ref:
                _v = {} if len(splits) > 0 else v
                ref[s] = _v
            ref = ref[s]
    return nested
