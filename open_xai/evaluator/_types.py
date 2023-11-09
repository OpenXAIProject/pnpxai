from dataclasses import dataclass
from typing import Dict

@dataclass
class EvaluatorOutput:
    explanation_results: dict
    evaluation_results: dict
    metrics_results: dict