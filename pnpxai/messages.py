from typing import Dict

MESSAGES: Dict[str, str] = {
    'experiment.event.explainer': 'Explaining with {explainer}.',
    'experiment.event.explainer.metric': 'Evaluating {metric} of {explainer}.',
    'experiment.errors.evaluation': 'Warning: Evaluating {metric_name} of {explainer_name} produced an error: {e}.',
    'experiment.errors.explanation': 'Warning: Explaining {explainer_name} produced an error: {e}.',
    'experiment.errors.explainer_unsupported': 'Warning: {explainer} is not currently supported.',
    'elapsed': 'Computed {task} in {elapsed} sec'
}


def get_message(key: str, *args, **kwargs):
    pattern = MESSAGES.get(key, None)
    return pattern.format(*args, **kwargs) if pattern is not None else key
