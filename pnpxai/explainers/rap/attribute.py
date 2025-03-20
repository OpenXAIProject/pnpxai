from typing import Any, Optional, Dict, Optional, List, Union, Callable

import torch
from torch import Tensor, nn

from pnpxai.core.detector.types import Linear, Convolution
from pnpxai.core._types import Model, DataSource, Task
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.rap.rap import RelativeAttributePropagation


class RAP(Explainer):
    """
    Computes Relative Attribute Propagation (RAP) explanations for a given model.

    Supported Modules: `Linear`, `Convolution`

    Parameters:
        model (Model): The model for which RAP explanations are computed.

    Reference:
        Woo-Jeoung Nam, Shir Gur, Jaesik Choi, Lior Wolf, Seong-Whan Lee. Relative Attributing Propagation: Interpreting the Comparative Contributions of Individual Units in Deep Neural Networks.
    """

    SUPPORTED_MODULES = [Linear, Convolution]
    SUPPORTED_DTYPES = [float]
    SUPPORTED_NDIMS = [4]
    alias = 'rap'

    def __init__(
        self,
        model: Model,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
    ):
        super().__init__(
            model,
            target_input_keys,
            additional_input_keys,
            output_modifier,
        )
        self.method = RelativeAttributePropagation(model)

    def compute_pred(self, output: Tensor) -> Tensor:
        """
        Computes the predicted class probabilities.

        Parameters:
            output (Tensor): The model output.

        Returns:
            Tensor: The one-hot encoded predicted class probabilities.
        """
        # get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]
        pred = pred.squeeze(-1)

        pred_one_hot = nn.functional.one_hot(pred, output.shape[-1]) * 1.0
        pred_one_hot = pred_one_hot.to(output.device)
        return pred_one_hot

    def attribute(self, inputs: DataSource, targets: DataSource, *args: Any, **kwargs: Any) -> DataSource:
        """
        Computes RAP attributions for the given inputs.

        Parameters:
            inputs (DataSource): The input data.
            targets (DataSource): The target labels.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            DataSource: RAP attributions.
        """
        target_inputs, _ = self.format_inputs(inputs)
        assert (
            len(target_inputs) == 1
        ), "RAP for multiple target inputs is not supported."
        outputs = self.method.run(*target_inputs)
        import pdb; pdb.set_trace()
        preds = self.compute_pred(outputs)
        relprop = self.method.relprop(preds)
        return relprop

    def format_outputs_for_visualization(
        self,
        inputs: DataSource,
        targets: DataSource,
        explanations: DataSource,
        task: Task,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        explanations = explanations.detach().cpu()\
            .transpose(-1, -3)\
            .transpose(-2, -3)

        explanations = explanations.sum(-1)
        agg_dims = list(range(1, explanations.ndim))
        pos_max = explanations.relu().amax(agg_dims, keepdim=True)
        neg_max = (-explanations).relu().amax(agg_dims, keepdim=True)

        explanations = torch.where(
            explanations > 0,
            explanations / pos_max,
            explanations / neg_max
        )

        return explanations.numpy()
