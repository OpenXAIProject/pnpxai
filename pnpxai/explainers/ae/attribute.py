"""PnPXAI Explainer interface for Attribution Equilibrium."""

from typing import Optional, Union

from torch import Tensor, nn

from pnpxai.core.detector.types import Convolution
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.types import ForwardArgumentExtractor
from pnpxai.explainers.ae.ae import AttributionEquilibrium


class AE(Explainer):
    """Explain an image classifier using Attribution Equilibrium.

    Parameters:
        model: PyTorch image classifier returning logits of shape (N, C).
        layer: Feature module, module path, or FX node name. Defaults follow
            the reference VGG/AlexNet/ResNet/DenseNet/Inception feature layers;
            other models use their last Conv2d. ResNet selects a block output.
        ratio: Positive relevance ratio in [1, 2], default 2.
        forward_arg_extractor: Extract the image tensor from a data batch.
        additional_forward_arg_extractor: Extract extra positional model args.

    Returns input-shaped signed relevance before channel pooling. Like the
    reference implementation, each image is explained on its own, so a
    result does not depend on the other images in its batch.

    Reference:
        Illuminating Salient Contributions in Neuron Activation with
        Attribution Equilibrium. DOI: 10.1109/TPAMI.2024.3485775.
    """

    SUPPORTED_MODULES = [Convolution]

    def __init__(
        self,
        model: nn.Module,
        layer: Optional[Union[str, nn.Module]] = None,
        ratio: float = 2.0,
        forward_arg_extractor: Optional[ForwardArgumentExtractor] = None,
        additional_forward_arg_extractor: Optional[ForwardArgumentExtractor] = None,
    ):
        super().__init__(model, forward_arg_extractor, additional_forward_arg_extractor)
        self.layer = layer
        self.ratio = ratio
        self.method = AttributionEquilibrium(self.model, layer=layer)

    def attribute(self, inputs: Tensor, targets=None) -> Tensor:
        """Return input-shaped signed AE relevance for the selected classes.

        Parameters:
            inputs (Tensor): A floating-point NCHW image batch, or a data batch
                from which the configured extractor obtains that image tensor.
            targets (Optional[Union[int, Tensor]]): One class index per image or
                a scalar to broadcast. None selects each predicted class.

        Returns:
            Tensor: Detached signed relevance with the image shape and device.
        """
        forward_args, additional_args = self._extract_forward_args(inputs)
        if isinstance(forward_args, tuple) and len(forward_args) == 1:
            forward_args = forward_args[0]
        if additional_args is None:
            additional_args = ()
        elif not isinstance(additional_args, tuple):
            additional_args = (additional_args,)
        # set_kwargs()/copy() can change the layer as well as ratio.
        if self.layer != self.method.original_layer:
            self.method = AttributionEquilibrium(self.model, layer=self.layer)
        return self.method.attribute(forward_args, targets, self.ratio, additional_args)

    def get_tunables(self):
        """Return the positive relevance ratio search space, float in [1, 2]."""
        return {"ratio": (float, {"low": 1.0, "high": 2.0})}
