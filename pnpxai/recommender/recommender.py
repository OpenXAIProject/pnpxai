from dataclasses import dataclass
import warnings
import torch.nn as nn

from pnpxai.explainers import *
from pnpxai.evaluator.infidelity import Infidelity
from pnpxai.evaluator.sensitivity import Sensitivity
from pnpxai.recommender._types import RecommenderOutput


class XaiRecommender:
    def __init__(self):
        self.question_table = {
            'why': {GuidedGradCam, Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, TCAV, Anchors},
            'how': {PDP},
            'why not': {CEM},
            'how to still be this': {Anchors},
        }
        self.task_table = {
            'image': {
                Lime, KernelShap,
                LRP, GuidedGradCam,
                # TODO: integrate RAP
                # RAP,
                # TODO: memory issue in IG, 
                # IntegratedGradients,
                # TODO: add more explainers
                # FullGrad, CEM, TCAV
            },
            'tabular': {Lime, KernelShap, PDP, CEM, Anchors},
            'text': {Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM},
        }
        self.architecture_table = {
            nn.Linear: {Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV, Anchors},
            nn.Conv1d: {GuidedGradCam, Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV, Anchors},
            nn.Conv2d: {GuidedGradCam, Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV, Anchors},
            nn.RNN: {Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV, Anchors},
            nn.Transformer: {Lime, KernelShap, IntegratedGradients, FullGrad, CEM, TCAV, Anchors},
            nn.MultiheadAttention: {Lime, KernelShap, IntegratedGradients, FullGrad, CEM, TCAV, Anchors},
        }
        self.evaluation_metric_table = {
            # Correctness -- Infidelity, Conitinuity -- Sensitivity
            GuidedGradCam: {Infidelity, Sensitivity},
            Lime: {Infidelity, Sensitivity},
            KernelShap: {Infidelity, Sensitivity},
            IntegratedGradients: {Infidelity, Sensitivity},
            FullGrad: {Infidelity, Sensitivity},
            LRP: {Infidelity, Sensitivity},
            RAP: {Infidelity, Sensitivity},

            # Evaluation metric not implemented yet
            PDP: {},
            CEM: {Infidelity, Sensitivity},
            TCAV: {Infidelity, Sensitivity},
            Anchors: {Infidelity, Sensitivity},
        }

    def _find_overlap(self, *lists):
        return list(set.intersection(*lists))

    def filter_methods(self, question, task, architecture):
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

        methods = self._find_overlap(
            question_to_method, task_to_method, architecture_to_method)

        return methods

    def suggest_metrics(self, methods):
        method_to_metric = [
            self.evaluation_metric_table[method]
            for method in methods if method in self.evaluation_metric_table
        ]
        metrics = self._find_overlap(*method_to_metric)
        return metrics

    def __call__(self, question, task, architecture):
        methods = self.filter_methods(question, task, architecture)
        metrics = self.suggest_metrics(methods)
        return RecommenderOutput(
            explainers=methods,
            evaluation_metrics=metrics,
        )
