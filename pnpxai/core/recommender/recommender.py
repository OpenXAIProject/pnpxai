from typing import List, Type, Dict, Set, Any, Sequence, Tuple
from tabulate import tabulate
from pnpxai.core._types import Modality, ExplanationType, Model
from pnpxai.core.detector import detect_model_architecture
from pnpxai.core.detector.types import (
    ModuleType,
    Linear,
    Convolution,
    LSTM,
    RNN,
    Attention,
    Embedding,
    SklearnModel,
)
from pnpxai.explainers.base import Explainer
from pnpxai.explainers import (
    GradCam,
    GuidedGradCam,
    Lime,
    KernelShap,
    Gradient,
    GradientXInput,
    SmoothGrad,
    VarGrad,
    IntegratedGradients,
    LRPUniformEpsilon,
    LRPEpsilonGammaBox,
    LRPEpsilonPlus,
    LRPEpsilonAlpha2Beta1,
    AttentionRollout,
    TransformerAttribution,
    TabLime,
    TabKernelShap,
)
from pnpxai.metrics.base import Metric
from pnpxai.metrics import (
    MuFidelity,
    Sensitivity,
    Complexity,
    TabABPC,
    TabMoRF,
    TabLeRF,
    TabAvgSensitivity,
    TabInfidelity,
    TabComplexity
)

from ._types import (
    RecommenderOutput,
)


# space, required, unsupported
DEFAULT_EXPLAINER_MAP = {
    GradCam: {
        "modalities": ["image"],
        "module_types": [Convolution],
    },
    GuidedGradCam: {
        "modalities": ["image"],
        "module_types": [Convolution],
    },
    Lime: {
        "modalities": ["image", "text", "time-series", "tabular", ("image", "text")],
        "module_types": [Linear, Convolution, LSTM, RNN, Attention],
    },
    KernelShap: {
        "modalities": ["image", "text", "time-series", "tabular", ("image", "text")],
        "module_types": [Linear, Convolution, LSTM, RNN, Attention],
    },
    Gradient: {
        "modalities": ["image", "text", ("image", "text")],
        "module_types": [Linear, Convolution, LSTM, RNN, Attention],
    },
    GradientXInput: {
        "modalities": ["image", "text", ("image", "text")],
        "module_types": [Linear, Convolution, LSTM, RNN, Attention],
    },
    SmoothGrad: {
        "modalities": ["text", ("image", "text")],
        "module_types": [Linear, Convolution, LSTM, RNN, Attention],
    },
    VarGrad: {
        "modalities": ["text", ("image", "text")],
        "module_types": [Linear, Convolution, LSTM, RNN, Attention],
    },
    IntegratedGradients: {
        "modalities": ["image", "text", ("image", "text"), "tabular"],
        "module_types": [Linear, Convolution, Attention],
    },
    LRPUniformEpsilon: {
        "modalities": ["image", "text", ("image", "text"), "tabular"],
        "module_types": [Linear, Convolution, LSTM, RNN, Attention],
    },
    LRPEpsilonGammaBox: {
        "modalities": ["image", "text", ("image", "text")],
        "module_types": [Convolution],
    },
    LRPEpsilonPlus: {
        "modalities": ["image", "text", ("image", "text")],
        "module_types": [Convolution],
    },
    LRPEpsilonAlpha2Beta1: {
        "modalities": ["image", "text", ("image", "text")],
        "module_types": [Convolution],
    },
    # AttentionRollout: {
    #     "modalities": ["text", ("image", "text")],
    #     "module_types": [Attention],
    # },
    # TransformerAttribution: {
    #     "modalities": ["text", ("image", "text")],
    #     "module_types": [Attention],
    # },
    TabLime: {
        'modalities': ['tabular'],
        'module_types': [SklearnModel],
    },
    TabKernelShap: {
        'modalities': ['tabular'],
        'module_types': [SklearnModel],
    }
}

DEFAULT_METRIC_MAP = {
    MuFidelity: {
        'modalities': ['image', 'text'],
    },
    Sensitivity: {
        'modalities': ['image', 'text'],
    },
    Complexity: {
        'modalities': ['image'],
    },
    TabABPC: {
        'modalities': ['tabular'],
    },
    TabMoRF: {
        'modalities': ['tabular'],
    },
    TabLeRF: {
        'modalities': ['tabular'],
    },
    TabAvgSensitivity: {
        'modalities': ['tabular'],
    },
    TabInfidelity: {
        'modalities': ['tabular'],
    },
    TabComplexity: {
        'modalities': ['tabular'],
    },
}


AVAILABLE_METRICS = {MuFidelity, Sensitivity, Complexity, TabABPC, TabMoRF, TabLeRF, TabAvgSensitivity, TabInfidelity, TabComplexity}

CAM_BASED_EXPLAINERS = {GradCam, GuidedGradCam}


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
        self.modality_to_explainers_map = None
        self.architecture_to_explainers_map = None
        self.explainer_to_metrics_map = None

        self._map_data = DEFAULT_EXPLAINER_MAP
        self._build_maps()

    def _build_maps(self):
        self._build_modality_to_explainers_map()
        self._build_architecture_to_explainers_map()
        self._build_explainer_to_metrics_map()

    def _build_modality_to_explainers_map(self):
        map_data = {}
        for explainer_type, v in self._map_data.items():
            for modality in v["modalities"]:
                if modality not in map_data:
                    map_data[modality] = set()
                map_data[modality].add(explainer_type)
        self.modality_to_explainers_map = RecommendationMap(map_data, ["modality", "explainers"])

    def _build_architecture_to_explainers_map(self):
        map_data = {}
        for explainer_type, v in self._map_data.items():
            for arch in v["module_types"]:
                if arch not in map_data:
                    map_data[arch] = set()
                map_data[arch].add(explainer_type)
        self.architecture_to_explainers_map = RecommendationMap(map_data, ["architecture", "explainers"])

    def _build_explainer_to_metrics_map(self):
        map_data = {}
        for explainer_type in self._map_data:
            for metric_type in AVAILABLE_METRICS:
                if explainer_type not in map_data:
                    map_data[explainer_type] = set()
                if explainer_type.EXPLANATION_TYPE == metric_type.SUPPORTED_EXPLANATION_TYPE:
                    map_data[explainer_type].add(metric_type)
        self.explainer_to_metrics_map = RecommendationMap(map_data, ["explainer", "metrics"])

    def add_explainer(
            self,
            explainer_type: Type[Explainer],
            supporting_modalities: List[Modality],
            supporting_module_types: List[ModuleType],
        ) -> None:
        if explainer_type in self._map_data:
            raise Exception(f"Explainer type '{explainer_type}' already exists.")
        self._map_data[explainer_type] = {
            "modalities": supporting_modalities,
            "module_types": supporting_module_types,
        }
        self._build_maps()

    def remove_explainer(self, explainer_type: Type[Explainer]) -> Dict[Type[Explainer], Dict[str, List]]:
        removed = self._map_data.pop(explainer_type, None)
        if removed is None:
            raise Exception(f"Explainer type '{explainer_type}' does not exist.")
        self._build_maps()
        return {explainer_type: removed}

    def update_explainer(
            self,
            explainer_type: Type[Explainer],
            supporting_modalities: List[Modality],
            supporting_module_types: List[ModuleType],
        ) -> None:
        if explainer_type not in self._map_data:
            raise Exception(f"Explainer type '{explainer_type}' does not exist.")
        self._map_data[explainer_type] = {
            "modalities": supporting_modalities,
            "module_types": supporting_module_types,
        }
        self._build_maps()

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
        modality_to_explainers = self.modality_to_explainers_map.data.get(modality, set())
        arch_to_explainers = set.union(*(
            self.architecture_to_explainers_map.data.get(module_type, set())
            for module_type in arch
        ))
        explainers = set.intersection(modality_to_explainers, arch_to_explainers)
        if arch.difference({Convolution, Linear}):
            explainers = explainers.difference(CAM_BASED_EXPLAINERS)
        return list(explainers)

    def _suggest_metrics(self, explainers: List[Type[Explainer]], modality: Modality) -> List[Type[Metric]]:
        """
        Suggests evaluation metrics based on the list of compatible explainers.

        Args:
        - methods (List[Type[Explainer]]): List of explainers supported for the given scenario.

        Returns:
        - List[Type[EvaluationMetric]]: List of compatible evaluation metrics for the explainers.
        """
        metrics = set.union(*(
            self.explainer_to_metrics_map.data.get(explainer, set())
            for explainer in explainers
        ))

        for metric in metrics.copy():
            if modality not in DEFAULT_METRIC_MAP[metric]['modalities']:
                metrics.remove(metric)

        return list(metrics)

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
        metrics = self._suggest_metrics(explainers, modality)
        return RecommenderOutput(
            architecture=arch,
            explainers=_sort_by_name(explainers),
            metrics=_sort_by_name(metrics),
        )




