from typing import Any, Optional, Dict

from torch import Tensor, nn

from pnpxai.core._types import Model, DataSource, Task
from pnpxai.explainers._explainer import Explainer
from pnpxai.explainers.rap.rap import RelativeAttributePropagation
from pnpxai.explainers.utils.post_process import postprocess_attr


class RAP(Explainer):
    def __init__(self, model: Model):
        super().__init__(model)
        self.method = RelativeAttributePropagation(model)
        self.device = next(self.model.parameters()).device

    def compute_pred(self, output):
        # get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]
        pred = pred.squeeze(-1)

        pred_one_hot = nn.functional.one_hot(pred, output.shape[-1]) * 1.0
        pred_one_hot = pred_one_hot.to(self.device)
        return pred_one_hot

    def attribute(self, inputs: DataSource, targets: DataSource, *args: Any, **kwargs: Any) -> DataSource:
        datum = inputs.to(self.device)
        outputs = self.model(datum)
        preds = self.compute_pred(outputs)
        relprop = self.method.relprop(preds, *args, **kwargs)
        return relprop

    def format_outputs_for_visualization(
        self,
        inputs: DataSource,
        targets: DataSource,
        explanations: DataSource,
        task: Task,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        explanations = explanations.transpose(-1, -3)\
            .transpose(-2, -3)
        return postprocess_attr(
            attr=explanations,
            sign="positive"
        )
