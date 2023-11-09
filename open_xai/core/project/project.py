from typing import Optional, List, Union, Sequence

from open_xai.detector import ModelArchitectureDetectorV2
from open_xai.recommender.recommender import XaiRecommender
from open_xai.evaluator import XaiEvaluator
from open_xai.explainers import ExplainerWArgs, Explainer
from open_xai.core.experiment import Experiment
from open_xai.core._types import Args, DataSource
from open_xai.core.project.auto_exp_input import AutoExplanationInput
from open_xai.explainers.config import default_attribute_kwargs
from open_xai.utils import class_to_string


class Project():
    def __init__(self, name: str):
        self.name = name
        self.experiments: List[Experiment] = []
        self.detector = ModelArchitectureDetectorV2()
        self.recommender = XaiRecommender()
        self.evaluator = XaiEvaluator()

    def auto_explain(self, exp_input: AutoExplanationInput):
        detector_output = self.detector(
            model=exp_input.model,
            sample=exp_input.data
        )

        recommender_output = self.recommender(
            question=exp_input.question,
            task=exp_input.task,
            architecture=detector_output.architecture,
        )

        explainers_w_args = [
            ExplainerWArgs(
                explainer(exp_input.model),
                Args(kwargs=default_attribute_kwargs.get(
                    class_to_string(explainer), {}
                ))
            )
            for explainer in recommender_output.explainers
        ]

        evaluator_metrics = [
            metric(default_attribute_kwargs.get(class_to_string(metric), {}))
            for metric in recommender_output.evaluation_metrics
        ]
        evaluator = XaiEvaluator(evaluator_metrics)

        inputs = [exp_input.input_extractor(batch) for batch in exp_input.data]
        labels = [exp_input.input_extractor(batch) for batch in exp_input.data]

        return self.explain(
            inputs,
            labels,
            explainers_w_args,
            evaluator
        )

    def explain(
        self,
        inputs: DataSource,
        outputs: DataSource,
        explainers: Sequence[Union[ExplainerWArgs, Explainer]],
        evaluator: Optional[XaiEvaluator] = None
    ) -> Experiment:
        experiment = Experiment(explainers, evaluator)
        experiment.run(inputs, outputs)
        self.experiments.append(experiment)

        return experiment

    def visualize(self):
        # TODO: Implement visualization technique
        for experiment in self.experiments:
            experiment.visualize()
