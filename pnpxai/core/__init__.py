# from .experiment import run_experiment
from pnpxai.core.detector.detector import detect_model_architecture
from pnpxai.core.recommender.recommender import XaiRecommender
from pnpxai.core.experiment import (
    Experiment,
    AutoExplanation,
    AutoExplanationForImageClassification,
    AutoExplanationForTextClassification,
    AutoExplanationForVisualQuestionAnswering,
    AutoExplanationForTabularClassification,    
)