from abc import abstractmethod


class EvaluationMetric():
    SUPPORTED_EXPLANATION_TYPE="attribution"

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
    
