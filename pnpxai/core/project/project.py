from typing import Optional, List, Union, Sequence

from pnpxai.detector import ModelArchitectureDetectorV2
from pnpxai.recommender.recommender import XaiRecommender
from pnpxai.evaluator import XaiEvaluator
from pnpxai.explainers import ExplainerWArgs, Explainer
from pnpxai.core.experiment import Experiment
from pnpxai.core._types import Args, DataSource
from pnpxai.core.project.auto_exp_input import AutoExplanationInput
from pnpxai.explainers.config import default_attribute_kwargs
from pnpxai.utils import class_to_string, CustomIterator


class Project():
    def __init__(self, name: str):
        self.name = name
        self.experiments: List[Experiment] = []
        self.detector = ModelArchitectureDetectorV2()
        self.recommender = XaiRecommender()

    def auto_explain(self, exp_input: AutoExplanationInput):
        inputs = exp_input.input_extractor(next(iter(exp_input.data)))

        detector_output = self.detector(
            model=exp_input.model,
            sample=inputs
        )

        recommender_output = self.recommender(
            question=exp_input.question,
            task=exp_input.task,
            architecture=detector_output.architecture,
        )

        explainers_w_args = [
            ExplainerWArgs(
                explainer(exp_input.model),
                Args(args=[], kwargs=default_attribute_kwargs.get(
                    class_to_string(explainer), {}
                ))
            )
            for explainer in recommender_output.explainers
        ]

        evaluator_metrics = [
            metric(**default_attribute_kwargs.get(class_to_string(metric), {}))
            for metric in recommender_output.evaluation_metrics
        ]
        evaluator = XaiEvaluator(evaluator_metrics)

        inputs = CustomIterator(exp_input.data, exp_input.input_extractor)
        labels = CustomIterator(exp_input.data, exp_input.label_extractor)

        return self.explain(
            inputs,
            labels,
            explainers_w_args,
            evaluator
        )

    def explain(
        self,
        inputs: DataSource,
        labels: DataSource,
        explainers: Sequence[Union[ExplainerWArgs, Explainer]],
        evaluator: Optional[XaiEvaluator] = None
    ) -> Experiment:
        explainers = [
            explainer if isinstance(explainer, ExplainerWArgs) else ExplainerWArgs(
                explainer,
                Args(args=[], kwargs=default_attribute_kwargs.get(
                    class_to_string(explainer), {}
                ))
            ) for explainer in explainers
        ]
        
        experiment = Experiment(explainers, evaluator)
        experiment.run(inputs, labels)
        self.experiments.append(experiment)

        return experiment

    def visualize(self):
        # TODO: Implement visualization technique
        for experiment in self.experiments:
            experiment.visualize()
