from typing import List, Type, Dict, Set, Any, Sequence, Tuple
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
from pnpxai.evaluator.metrics.base import Metric
from pnpxai.evaluator.metrics import (
    MuFidelity,
    Sensitivity,
    Complexity,
    MoRF,
    LeRF,
    AbPC,
    AVAILABLE_METRICS,
)


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
        modality: Modality,
        arch: Set[ModuleType],
    ) -> List[Type[Explainer]]:
        """
        Filters explainers based on the user's question, task, and model architecture.

        Args:
        - question (str): User's question type ('why', 'how', 'why not', 'how to still be this').
        - task (str): Task type ('image', 'tabular', 'text').
        - architecture (List[Type]): List of neural network module types (e.g., nn.Linear, nn.Conv2d).

        Returns:
        - List[Type[Explainer]]: List of compatible explainers based on the given inputs.
        """
        # question_to_method = QUESTION_TO_EXPLAINERS.get(question, set())
        modality_to_explainers = set(modality.EXPLAINERS)
        arch_to_explainers = set.union(*(
            self.architecture_to_explainers_map.data.get(module_type, set())
            for module_type in arch
        ))
        explainers = set.intersection(
            modality_to_explainers, arch_to_explainers)
        if arch.difference({Convolution, Linear}):
            explainers = explainers.difference(CAM_BASED_EXPLAINERS)
        return list(explainers)

    # def _suggest_metrics(self, explainers: List[Type[Explainer]]):
    #     """
    #     Suggests evaluation metrics based on the list of compatible explainers.

    #     Args:
    #     - methods (List[Type[Explainer]]): List of explainers supported for the given scenario.

    #     Returns:
    #     - List[Type[EvaluationMetric]]: List of compatible evaluation metrics for the explainers.
    #     """
    #     metrics = set.union(*(
    #         self.explainer_to_metrics_map.data.get(explainer, set())
    #         for explainer in explainers
    #     ))
    #     return list(metrics)

    def recommend(self, modality: Modality, model: Model):
        """
        Recommends explainers and evaluation metrics based on the user's input.

        Args:
        - question (str): User's question type ('why', 'how', 'why not', 'how to still be this').
        - task (str): Task type ('image', 'tabular', 'text').
        - architecture (List[Type]): List of neural network module types (e.g., nn.Linear, nn.Conv2d).

        Returns:
        - RecommenderOutput: An object containing recommended explainers and evaluation metrics.
        """
        arch = detect_model_architecture(model)
        explainers = self._filter_explainers(modality, arch)
        # metrics = self._suggest_metrics(explainers)
        return RecommenderOutput(
            detected_architectures=arch,
            explainers=_sort_by_name(explainers),
            # metrics=_sort_by_name(metrics),
        )
