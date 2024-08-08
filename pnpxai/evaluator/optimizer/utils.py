from typing import Literal, Dict, Any

import optuna
from pnpxai.explainers.utils.postprocess import RELEVANCE_POOLING_METHODS
from pnpxai.explainers.utils.baselines import DEFAULT_BASELINE_METHODS


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

def format_params(params: Dict[str, Any]):
    formatted = params.copy()
    for k in params.keys():
        if k == 'baseline_fn':
            formatted[k] = DEFAULT_BASELINE_METHODS[k]
    return formatted