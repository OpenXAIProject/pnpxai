from xai_pnp.core._types import Model
from xai_pnp.explainers._explainer import Explainer
from xai_pnp.core.experiment import Experiment


class Project():
    def __init__(self, name: str):
        self.name = name
        self.experiments = []

    def make_experiment(self, model: Model, explainer: Explainer) -> Experiment:
        experiment = Experiment(model, explainer)
        self.experiments.append(experiment)

        return experiment

    def visualize(self):
        # TODO: Implement visualization technique
        for experiment in self.experiments:
            experiment.visualize()