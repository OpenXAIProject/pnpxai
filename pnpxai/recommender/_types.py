from dataclasses import dataclass

@dataclass
class RecommenderOutput:
    explainers: list
    evaluation_metrics: list