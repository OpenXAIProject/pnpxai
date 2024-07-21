from io import TextIOWrapper
from typing import Optional, Union, Sequence, TypeVar, Dict, Type
import yaml

from pnpxai.messages import get_message
from pnpxai.utils import open_file_or_name
from pnpxai.core._types import ConfigKeys, ModalityOrListOfModalities, Model
# from pnpxai.core.experiment.experiment_explainer_defaults import EXPLAINER_AUTO_KWARGS
# from pnpxai.core.experiment.experiment_metrics_defaults import EVALUATION_METRIC_AUTO_KWARGS
from pnpxai.explainers.base import Explainer
# from pnpxai.explainers import AVAILABLE_EXPLAINERS, ExplainerWArgs, Explainer
from pnpxai.metrics.base import Metric
# from pnpxai.evaluator import AVAILABLE_METRICS, Metric


T = TypeVar('T')


def _get_name_to_value_map(values: Sequence[T]) -> Dict[str, T]:
    return {
        str(value.__name__): value
        for value in values
    }


AVAILABLE_EXPLAINERS_MAP = _get_name_to_value_map(AVAILABLE_EXPLAINERS)
AVAILABLE_METRICS_MAP = _get_name_to_value_map(AVAILABLE_METRICS)


class ProjectConfig:
    def __init__(
        self,
        modality: Optional[ModalityOrListOfModalities] = None,
        explainers: Optional[Sequence[Union[
            Explainer, Type[Explainer], str
        ]]] = None,
        metrics: Optional[Sequence[Union[Type[Metric], str]]] = None,
    ):
        self._modality = modality
        self._explainers = [] if explainers is None else explainers
        self._metrics = [] if metrics is None else metrics

    @classmethod
    def _parse_config(cls, config: Optional[Union[dict, str, TextIOWrapper]] = None):
        if config is None or isinstance(config, dict):
            return config

        if not isinstance(config, (str, TextIOWrapper)):
            raise Exception(get_message(
                'project.config.unsupported',
                config_type=type(config)
            ))

        with open_file_or_name(config, 'r') as config_file:
            return yaml.safe_load(config_file)

    @classmethod
    def from_predefined(cls, config: Optional[Union[dict, str, TextIOWrapper]] = None) -> dict:
        config = cls._parse_config(config) or {}
        return cls(
            modality=config.get(ConfigKeys.modality.value, None),
            explainers=config.get(ConfigKeys.EXPLAINERS.value, None),
            metrics=config.get(ConfigKeys.METRICS.value, None),
        )

    def get_modality(self) -> ModalityOrListOfModalities:
        return self._modality

    # def get_init_explainers(self, model: Model):
    #     explainer_types = []
    #     for explainer in self._explainers:
    #         if explainer is None:
    #             continue

    #         if isinstance(explainer, str):
    #             explainer = AVAILABLE_EXPLAINERS_MAP.get(explainer, None)

    #         if isinstance(explainer, type(Explainer)):
    #             explainer = explainer(model)

    #         if isinstance(explainer, Explainer):
    #             explainer = ExplainerWArgs(
    #                 explainer=explainer,
    #                 kwargs=EXPLAINER_AUTO_KWARGS.get(type(explainer), None)
    #             )

    #         if isinstance(explainer, ExplainerWArgs):
    #             explainer_types.append(explainer)

    #     return explainer_types

    # def get_init_metrics(self,):
    #     metric_types = []
    #     for metric in self._metrics:
    #         if metric is None:
    #             continue

    #         if isinstance(metric, str):
    #             metric = AVAILABLE_METRICS_MAP.get(metric, None)

    #         if isinstance(metric, type(Metric)):
    #             metric = metric(
    #                 **EVALUATION_METRIC_AUTO_KWARGS.get(type(metric), {})
    #             )

    #         if isinstance(metric, Metric):
    #             metric_types.append(metric)

    #     return metric_types
