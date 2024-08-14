# from .experiment import run_experiment
from .detector import detect_model_architecture
from .recommender import XaiRecommender
from .experiment import (
    Experiment,
    AutoExplanation,
    AutoExplanationForImageClassification,
    AutoExplanationForTextClassification,
    AutoExplanationForVisualQuestionAnswering,
    AutoExplanationForTabularClassification,    
)