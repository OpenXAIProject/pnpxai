from typing import List, Type, Dict, Set, Any, Sequence, Tuple, Union
from dataclasses import dataclass, asdict
from tabulate import tabulate

from pnpxai.core._types import Model
from pnpxai.core.modality.modality import Modality
from pnpxai.core.detector import detect_model_architecture
from pnpxai.core.detector.types import (
    ModuleType,
    Linear,
    Convolution
)
from pnpxai.explainers.base import Explainer
from pnpxai.explainers import (
    GradCam,
    GuidedGradCam,
    AVAILABLE_EXPLAINERS
)
from pnpxai.utils import format_into_tuple


CAM_BASED_EXPLAINERS = {GradCam, GuidedGradCam}


@dataclass
class RecommenderOutput:
    detected_architectures: Set[ModuleType]
    explainers: list

    def print_tabular(self):
        print(tabulate([
            [k, [v.__name__ for v in vs]]
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
        self.architecture_to_explainers_map = self._build_architecture_to_explainers_map()

    def _build_architecture_to_explainers_map(self):
        map_data = {}
        for explainer_type in AVAILABLE_EXPLAINERS:
            for arch in explainer_type.SUPPORTED_MODULES:
                if arch not in map_data:
                    map_data[arch] = set()
                map_data[arch].add(explainer_type)
        return RecommendationMap(map_data, ["architecture", "explainers"])

    def _filter_explainers(
        self,
        modality: Union[Modality, Tuple[Modality]],
        arch: Set[ModuleType],
    ) -> List[Type[Explainer]]:
        """
        Filters explainers based on the user's question, task, and model architecture.

        Args:
        - modaltiy (Union[Modality, Tuple[Modality]]): Modality of the input data (e.g., ImageModality, TextModality, TabularModality).
        - arch (Set[ModuleType]): Set of neural network module types (e.g., nn.Linear, nn.Conv2d).

        Returns:
        - List[Set[Type[Explainer]]]: List of compatible explainers based on the given inputs.
        """
        # question_to_method = QUESTION_TO_EXPLAINERS.get(question, set())
        explainers = []
        for mod in format_into_tuple(modality):
            modality_to_explainers = set(mod.EXPLAINERS)
            arch_to_explainers = set.union(*(
                self.architecture_to_explainers_map.data.get(
                    module_type, set())
                for module_type in arch
            ))
            explainers_mod = set.intersection(
                modality_to_explainers, arch_to_explainers)
            if arch.difference({Convolution, Linear}):
                explainers_mod = explainers_mod.difference(
                    CAM_BASED_EXPLAINERS)
            explainers.append(explainers_mod)
        explainers = set.intersection(*explainers)
        return list(explainers)

    def recommend(self, modality: Union[Modality, Tuple[Modality]], model: Model) -> RecommenderOutput:
        """
        Recommends explainers and evaluation metrics based on the user's input.

        Args:
        - modality (Union[Modality], Tuple[Modality]): Modality of the input data (e.g., ImageModality, TextModality, TabularModality).
        - model (Model): Neural network module, used for the architecture-based filtering.

        Returns:
        - RecommenderOutput: An object containing recommended explainers.
        """
        arch = detect_model_architecture(model)
        explainers = self._filter_explainers(modality, arch)
        # metrics = self._suggest_metrics(explainers)
        return RecommenderOutput(
            detected_architectures=arch,
            explainers=_sort_by_name(explainers),
            # metrics=_sort_by_name(metrics),
        )
