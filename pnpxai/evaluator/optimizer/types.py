from dataclasses import dataclass

import optuna
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils.postprocess import PostProcessor


@dataclass
class OptimizationOutput:
    explainer: Explainer
    postprocessor: PostProcessor
    study: optuna.study.Study
