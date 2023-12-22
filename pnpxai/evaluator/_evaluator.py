from abc import abstractmethod
from pnpxai.explainers import ExplainerWArgs
from pnpxai.core._types import Model, DataSource, TensorOrTensorSequence


class EvaluationMetric():
    @abstractmethod
    def __call__(
        self,
        model: Model,
        explainer_w_args: ExplainerWArgs,
        inputs: DataSource,
        targets: DataSource,
        explanations: TensorOrTensorSequence
    ):
        raise NotImplementedError()
