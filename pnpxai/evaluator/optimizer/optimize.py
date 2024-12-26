from typing import Literal, Optional
import torch
import optuna
from pnpxai.evaluator.optimizer.objectives import Objective
from pnpxai.evaluator.optimizer.utils import (
    load_sampler,
    get_default_n_trials,
)

def optimize(
    objective: Objective,
    direction: Literal['maximize', 'minimize'] = 'maximize',
    sampler: Literal['random', 'tpe'] = 'random',
    n_trials: Optional[int] = None,
    **kwargs
) -> optuna.study.Study:
    """
    Optimize hyperparameters by processing data, generating explanations, evaluating with metrics, caching and retrieving the data.

    Args:
        objective (Objective): An instance of ``pnpxai.evaluator.optimizer.Objective`` which defines explainer, postprocessor, metric, and data
        direction (Literal['minimize', 'maximize']): A string to specify the direction of optimization.
        sampler (Literal['grid', 'random', 'tpe']): A string to specify the sampler to use for optimization.
        n_trials (Optional[int]): An integer to specify the number of trials for optimization. If none passed, the number of trials is inferred from `timeout`.
        **kwargs: kwargs for any sampler in ``optuna.samplers``.

    Returns:
        The instance of ``optuna.study.Study`` containing explainers and postprocessor for each trial.
    """

    study = optuna.create_study(
        sampler=load_sampler(sampler, **kwargs),
        direction=direction,
    )
    n_trials = n_trials or get_default_n_trials(sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
    )
    return study
