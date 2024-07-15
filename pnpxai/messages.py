from typing import Dict

MESSAGES: Dict[str, str] = {
    'experiment.event.explainer': 'Explaining with {explainer}.',
    'experiment.event.explainer.metric': 'Evaluating {metric} of {explainer}.',
    'experiment.errors.evaluation': 'Warning: Evaluating {metric} of {explainer} produced an error: {error}.',
    'experiment.errors.explanation': 'Warning: Explaining {explainer} produced an error: {error}.',
    'experiment.errors.explainer_unsupported': 'Warning: {explainer} is not currently supported.',
    'elapsed': 'Computed {modality} in {elapsed} sec',
    'project.config.unsupported': 'Error: Config of type {config_type} is not supported'
}


def get_message(key: str, *args, **kwargs):
    pattern = MESSAGES.get(key, None)
    return pattern.format(*args, **kwargs) if pattern is not None else key
