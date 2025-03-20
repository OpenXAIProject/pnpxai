from typing import List, Type, Dict, Set, Any, Sequence, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass, asdict
from tabulate import tabulate
import itertools

from pnpxai.core._types import Model
from pnpxai.core.modality.modality import Modality
from pnpxai.core.detector import (
    detect_model_architecture,
    detect_data_modality,
)
from pnpxai.core.detector.detector import _data_modality_maybe
from pnpxai.core.detector.types import ModuleType, Attention
from pnpxai.explainers.base import Explainer
from pnpxai.explainers import (
    GradCam,
    GuidedGradCam,
    RAP,
    AVAILABLE_EXPLAINERS
)
from pnpxai.utils import format_into_tuple


ATTENTION_NOT_SUPPORTED_EXPLAINERS = {GradCam, GuidedGradCam, RAP}


@dataclass
class RecommenderOutput:
    detected_modality: str
    detected_architectures: Set[ModuleType]
    explainers: List[Explainer]

    def print_tabular(self):
        print(tabulate([
            [k, [v.__name__ if not isinstance(v, str) else v for v in vs]]
            for k, vs in asdict(self).items()
        ]))


def _sort_by_name(vals):
    return list(sorted(vals, key=lambda x: x.__name__))


class RecommendationMap:
    def __init__(self, data: Dict[Any, Set[Any]], headers: Sequence[str]) -> None:
        self.data = data
        self.headers = headers

    def print_tabular(self):
        table = [self.headers]
        for k, vs in self.data.items():
            if not isinstance(k, (str, Tuple)):
                k = k.__name__
            table.append([k, ', '.join(sorted(v.__name__ for v in vs))])
        print(tabulate(table, headers="firstrow"))


class XaiRecommender:
    """
    Recommender class that suggests explainers and evaluation metrics based on the user's input. The recommender
    analyzes the model architecture and the selected modality to provide the most suitable explainers and detected architecture.        
    """

    def __init__(self):
        self._map_by_architecture = self._build_map_by_architecture()
        self._map_by_modality = self._build_map_by_modality()

    @property
    def map_by_architecture(self):
        return self._map_by_architecture
    
    @property
    def map_by_modality(self):
        return self._map_by_modality

    def _build_map_by_architecture(self):
        map_data = defaultdict(set)
        for explainer_type in AVAILABLE_EXPLAINERS:
            for arch in explainer_type.SUPPORTED_MODULES:
                map_data[arch].add(explainer_type)
        return RecommendationMap(map_data, ["architecture", "explainers"])

    def _build_map_by_modality(self):
        map_data = defaultdict(set)
        for explainer_type in AVAILABLE_EXPLAINERS:
            if not hasattr(explainer_type, 'SUPPORTED_DTYPES'):
                continue
            combs = itertools.product(
                explainer_type.SUPPORTED_DTYPES,
                explainer_type.SUPPORTED_NDIMS,
            )
            for comb in combs:
                mod_nm = _data_modality_maybe(*comb)
                if not mod_nm:
                    continue
                map_data[mod_nm].add(explainer_type)
        return RecommendationMap(map_data, ["modality", "explainers"])

    def _filter_explainers(
        self,
        modality: Union[Modality, Tuple[Modality]],
        architecture: Set[ModuleType],
    ) -> List[Type[Explainer]]:
        """
        Filters explainers based on the user's question, task, and model architecture.

        Args:
        - modaltiy (Union[Modality, Tuple[Modality]]): Modality of the input data (e.g., ImageModality, TextModality, TabularModality).
        - architecture (Set[ModuleType]): Set of neural network module types (e.g., nn.Linear, nn.Conv2d).

        Returns:
        - List[Set[Type[Explainer]]]: List of compatible explainers based on the given inputs.
        """
        # question_to_method = QUESTION_TO_EXPLAINERS.get(question, set())
        explainers = defaultdict(set)
        explainers['modality'].update(AVAILABLE_EXPLAINERS)
        for mod in format_into_tuple(modality):
            mod_nm = _data_modality_maybe(mod.dtype_key, mod.ndims)
            if mod_nm is None:
                raise ValueError('Cannot match data modality')
            explainers['modality'].difference_update(
                explainers['modality'].difference(
                    self._map_by_modality.data[mod_nm]
                )
            )
        for arch in architecture:
            if arch in self._map_by_architecture.data:
                explainers['architecture'].update(
                    self._map_by_architecture.data[arch]
                )
        if Attention in architecture:
            explainers['architecture'].difference_update(ATTENTION_NOT_SUPPORTED_EXPLAINERS)
        explainers = set.intersection(*explainers.values())
        return list(explainers)

    def recommend(
        self,
        modality: Union[Modality, Tuple[Modality]],
        model: Model,
    ) -> RecommenderOutput:
        """
        Recommends explainers and evaluation metrics based on the user's input.

        Args:
        - modality (Union[Modality], Tuple[Modality]): Modality of the input data (e.g., ImageModality, TextModality, TabularModality).
        - model (Model): Neural network module, used for the architecture-based filtering.

        Returns:
        - RecommenderOutput: An object containing recommended explainers.
        """
        mod_nms = detect_data_modality(modality)
        architecture = detect_model_architecture(model)
        explainers = self._filter_explainers(modality, architecture)
        return RecommenderOutput(
            detected_modality=mod_nms,
            detected_architectures=architecture,
            explainers=_sort_by_name(explainers),
        )
