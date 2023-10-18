from typing import Optional, List

from open_xai.detector.detector import Detector
from open_xai.core._types import Model
from open_xai.explainers._explainer import Explainer
from open_xai.core.experiment import Experiment


class Project():
    def __init__(self, name: str):
        self.name = name
        self.experiments: List[Experiment] = []
        self.detector = Detector()

    def auto_explain(self, model: Model):
        explainer = self.detector(model)
        return self.explain(explainer)

    def explain(self, explainer: Explainer) -> Experiment:
        experiment = Experiment(explainer)
        self.experiments.append(experiment)

        return experiment

    def visualize(self):
        # TODO: Implement visualization technique
        for experiment in self.experiments:
            experiment.visualize()
