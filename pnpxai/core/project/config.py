from io import TextIOWrapper
from typing import Optional, Union, Sequence
import yaml

from pnpxai.messages import get_message
from pnpxai.utils import open_file_or_name, class_to_string
from pnpxai.core._types import ConfigKeys, Task, Model
from pnpxai.core.experiment.utils import init_explainers, init_metrics
from pnpxai.explainers import AVAILABLE_EXPLAINERS
from pnpxai.evaluator import AVAILABLE_METRICS


def _get_name_to_value_map(values: Sequence):
    return {
        str(value.__name__): value
        for value in values
    }


AVAILABLE_EXPLAINERS_MAP = _get_name_to_value_map(AVAILABLE_EXPLAINERS)
AVAILABLE_METRICS_MAP = _get_name_to_value_map(AVAILABLE_METRICS)


class ProjectConfig:
    def __init__(self, config: Optional[Union[dict, str, TextIOWrapper]] = None):
        self._config = self._parse_config(config)

    def _parse_config(self, config: Optional[Union[dict, str, TextIOWrapper]] = None) -> dict:
        if config is None or isinstance(config, dict):
            return config

        if not isinstance(config, (str, TextIOWrapper)):
            raise Exception(get_message(
                'project.config.unsupported',
                config_type=type(config)
            ))

        with open_file_or_name(config, 'r') as config_file:
            return yaml.safe_load(config_file)
    
    def get_task(self) -> Task:
        return self._config.get(ConfigKeys.TASK.value, None)

    def get_init_explainers(self, model: Model):
        explainers = self._config.get(ConfigKeys.EXPLAINERS.value, [])
        explainer_types = filter(None, [
            AVAILABLE_EXPLAINERS_MAP.get(explainer, None)
            for explainer in explainers
        ])

        return init_explainers(model, explainer_types)

    def get_init_metrics(self,):
        metrics = self._config.get(ConfigKeys.METRICS.value, [])
        metric_types = filter(None, [
            AVAILABLE_METRICS_MAP.get(metric, None)
            for metric in metrics
        ])

        return init_metrics(metric_types)
