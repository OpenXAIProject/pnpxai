import warnings
import torch.nn as nn
from typing import List, Type

from pnpxai.explainers import *
from pnpxai.evaluator.mu_fidelity import MuFidelity
from pnpxai.evaluator.sensitivity import Sensitivity
from pnpxai.evaluator.complexity import Complexity
from pnpxai.recommender._types import RecommenderOutput


class XaiRecommender:
    """
    Recommends explainability methods and associated evaluation metrics based on user's question, task, and model architecture.

    Attributes:
        question_table (dict): Maps questions ('why', 'how', 'why not', 'how to still be this') to supported explainers.
        task_table (dict): Maps tasks ('image', 'tabular', 'text') to supported explainers.
        architecture_table (dict): Maps neural network module types (e.g., nn.Linear, nn.Conv2d) to supported explainers.
        evaluation_metric_table (dict): Maps explainers to supported evaluation metrics (MuFidelity, Sensitivity, Complexity).
    """
        def __init__(self):
        self.question_table = {
            'why': {
                GuidedGradCam, Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, TCAV, Anchors
                # ,GradCam
                },
            'how': {PDP},
            'why not': {CEM},
            'how to still be this': {Anchors},
        }
        self.task_table = {
            'image': {
                GuidedGradCam, Lime, KernelShap, LRP, RAP, IntegratedGradients,
                # GradCam
                # TODO: add more explainers
                # FullGrad, CEM, TCAV
            },
            'tabular': {Lime, KernelShap, PDP, CEM, Anchors},
            'text': {
                # Lime, KernelShap,
                IntegratedGradients, FullGrad, LRP, RAP, CEM},
        }
        self.architecture_table = {
            nn.Linear: {
                Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV, Anchors},
            nn.Conv1d: {
                # GradCam,
                GuidedGradCam, Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV, Anchors},
            nn.Conv2d: {
                # GradCam,
                GuidedGradCam, Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV, Anchors},
            nn.RNN: {
                Lime, KernelShap,
                IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV, Anchors},
            nn.Transformer: {
                Lime, KernelShap, LRP,
                IntegratedGradients, FullGrad, CEM, TCAV, Anchors},
            nn.MultiheadAttention: {
                Lime, KernelShap, LRP,
                IntegratedGradients, FullGrad, CEM, TCAV, Anchors},
        }
        self.evaluation_metric_table = {
            # Correctness -- MuFidelity, Conitinuity -- Sensitivity, Compactness -- Complexity
            # GradCam: {MuFidelity, Sensitivity, Complexity},
            GuidedGradCam: {MuFidelity, Sensitivity, Complexity},
            Lime: {MuFidelity, Sensitivity, Complexity},
            KernelShap: {MuFidelity, Sensitivity, Complexity},
            IntegratedGradients: {MuFidelity, Sensitivity, Complexity},
            FullGrad: {MuFidelity, Sensitivity, Complexity},
            LRP: {MuFidelity, Sensitivity, Complexity},
            RAP: {MuFidelity, Sensitivity, Complexity},

            # Evaluation metric not implemented yet
            PDP: {},
            CEM: {MuFidelity, Sensitivity},
            TCAV: {MuFidelity, Sensitivity},
            Anchors: {MuFidelity, Sensitivity},
        }

    def _find_overlap(self, *sets):
        """
        Finds the unique intersection of any number of sets.
        This function is specifically used to deduplicate elements across multiple sets during the filtering process.
        - For explainers: It removes instances where an explainer is supported by multiple criteria (e.g., question, task, and architecture),
          ensuring only one unique recommendation per explainer.
        - For evaluation metrics: It eliminates redundant suggestions arising from overlaps between compatible explainers and their supported metrics.

        Args:
        - *sets: Sets to perform intersection on.

        Returns:
        - List: The list of unique elements present in all sets.
        """
        sets = sets or [set()]
        return list(set.intersection(*sets))

    def filter_methods(self, question, task, architecture) -> List[Type[Explainer]]:
        """
        Filters explainers based on the user's question, task, and model architecture.

        Args:
        - question (str): User's question type ('why', 'how', 'why not', 'how to still be this').
        - task (str): Task type ('image', 'tabular', 'text').
        - architecture (List[Type]): List of neural network module types (e.g., nn.Linear, nn.Conv2d).

        Returns:
        - List[Type[Explainer]]: List of compatible explainers based on the given inputs.
        """
        question_to_method = self.question_table[question]
        task_to_method = self.task_table[task]

        architecture_to_method = []
        for module in architecture:
            try:
                architecture_to_method.append(self.architecture_table[module])
            except KeyError:
                warnings.warn(
                    f"\n[Recommender] Warning: {repr(module)} is not currently supported.")

        architecture_to_method = self._find_overlap(*architecture_to_method)
        if (nn.Conv1d in architecture or nn.Conv2d in architecture) and GuidedGradCam not in architecture_to_method:
            if nn.MultiheadAttention not in architecture:
                architecture_to_method.append(GuidedGradCam)
                architecture_to_method.append(GradCam)

        methods = self._find_overlap(
            question_to_method, task_to_method, architecture_to_method)

        return methods

    def suggest_metrics(self, methods):
        """
        Suggests evaluation metrics based on the list of compatible explainers.

        Args:
        - methods (List[Type[Explainer]]): List of explainers supported for the given scenario.

        Returns:
        - List[Type[EvaluationMetric]]: List of compatible evaluation metrics for the explainers.
        """
        method_to_metric = [
            self.evaluation_metric_table[method]
            for method in methods if method in self.evaluation_metric_table
        ]
        metrics = self._find_overlap(*method_to_metric)
        return metrics

    def _sort_by_name(self, vals):
        """
        Sorts a list of values by their names.

        Args:
        - vals (List): List of values to sort.

        Returns:
        - List: Sorted list of values based on their names.
        """
        return list(sorted(vals, key=lambda x: x.__name__))

    def __call__(self, question, task, architecture):
        """
        Recommends explainers and evaluation metrics based on the user's input.

        Args:
        - question (str): User's question type ('why', 'how', 'why not', 'how to still be this').
        - task (str): Task type ('image', 'tabular', 'text').
        - architecture (List[Type]): List of neural network module types (e.g., nn.Linear, nn.Conv2d).

        Returns:
        - RecommenderOutput: An object containing recommended explainers and evaluation metrics.
        """
        methods = self.filter_methods(question, task, architecture)
        methods = self._sort_by_name(methods)
        metrics = self.suggest_metrics(methods)
        metrics = self._sort_by_name(metrics)
        return RecommenderOutput(
            explainers=methods,
            evaluation_metrics=metrics,
        )
