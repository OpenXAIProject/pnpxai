from abc import abstractmethod


class EvaluationMetric():
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
    
