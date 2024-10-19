#%%
import optuna
from optuna.samplers import RandomSampler

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(sampler=RandomSampler())
study.optimize(objective, n_trials=1000)

study.best_params  # E.g. {'x': 2.002108042}
# %%
sorted_trials = sorted(
    [trial for trial in study.trials if trial.value is not None],
    key=lambda trial: trial.value,
)
# %%
sorted_trials[-1]
#%%
sorted_trials[-1].__dict__
# %%
