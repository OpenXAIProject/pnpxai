from typing import Literal

import optuna


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
