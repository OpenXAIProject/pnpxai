import warnings
from typing import List, Any, Dict, Type, Literal

from time import time_ns
from pnpxai.detector import ModelArchitectureDetector
from pnpxai.explainers import Explainer
from pnpxai.recommender.recommender import XaiRecommender
from pnpxai.evaluator import XaiEvaluator
from pnpxai.core._types import DataSource, Model
from pnpxai.core.experiment.run import Run


class Experiment:
    def __init__(
        self,
        name: str,
        model: Model,
        evaluator: XaiEvaluator,
        task: Literal["image", "tabular"] = "image",
    ):
        self.name = name
        self.model = model
        self.evaluator = evaluator

        self.explainers: List[Explainer] = []
        self.task = task

        self.runs: List[Run] = []
    
    def __repr__(self):
        return f"<Experiment: {self.name}>"
        
    def add_explainer(
            self,
            explainer_type: Type[Explainer],
            additional_kwargs: Dict[str, Any] = {},
        ):
        if explainer_type not in self.list_applicable_explainers():
            warnings.warn(f"{explainer} is not applicable for {self.model}")
        explainer = explainer_type(self.model)
        for k, v in additional_kwargs:
            explainer.additional_kwargs[k] = additional_kwargs.pop(k)
        self.explainers.append(explainer)
    
    def remove_explainer(self, idx: int):
        removed = self.explainers.pop(idx)
        return removed
    
    def auto_add_explainers(
            self,
            question: Literal["why", "how"] = "why"
        ):
        # if auto_add, set default kwargs
        for explainer_type in self.list_applicable_explainers(question):
            self.add_explainer(explainer_type)
    
    def list_applicable_explainers(
            self,
            question: Literal["why", "how"] = "why",
        ) -> List[Type[Explainer]]:        
        # TODO: no need to be a class
        detector = ModelArchitectureDetector()
        architecture = detector(self.model, sample=None).architecture # TODO: no sample needed

        # TODO: no need to be a class
        recommender = XaiRecommender()
        applicables = recommender(
            question = question,
            task = self.task,
            architecture = architecture
        ).explainers
        return applicables
    
    def _add_run(self, run: Run):
        self.runs.append(run)

    def run(
            self,
            inputs: DataSource,
            targets: DataSource,
        ) -> 'Experiment':
        for explainer in self.explainers:
            run = Run(
                inputs = inputs,
                targets = targets,
                explainer = explainer,
                evaluator = self.evaluator,
            )
            run.execute()
            self._add_run(run)
        return self