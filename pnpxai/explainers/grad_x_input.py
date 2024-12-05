from typing import Callable, Optional, List, Tuple, Union, Sequence

from torch import Tensor
from torch.nn.modules import Module
from captum.attr import InputXGradient as CaptumGradientXInput
from captum.attr import LayerGradientXActivation as CaptumLayerGradientXInput

from pnpxai.core.detector.types import Linear, Convolution, LSTM, RNN, Attention
from .base import Explainer
from .utils import captum_wrap_model_input


class GradientXInput(Explainer):
    SUPPORTED_MODULES = [Linear, Convolution, LSTM, RNN, Attention]

    def __init__(
        self,
        model: Module,
        layer: Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]] = None,
        forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        additional_forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
    ) -> None:
        super().__init__(model, forward_arg_extractor, additional_forward_arg_extractor)
        self.layer = layer


    @property
    def _layer_explainer(self) -> CaptumLayerGradientXInput:
        wrapped_model = captum_wrap_model_input(self.model)
        layers = [
            wrapped_model.input_maps[layer] if isinstance(layer, str)
            else layer for layer in self.layer
        ] if isinstance(self.layer, Sequence) else self.layer
        return CaptumLayerGradientXInput(
            forward_func=wrapped_model,
            layer=layers,
        )

    @property
    def _explainer(self) -> CaptumGradientXInput:
        return CaptumGradientXInput(forward_func=self.model)
    
    @property
    def explainer(self) -> Union[CaptumGradientXInput, CaptumLayerGradientXInput]:
        if self.layer is None:
            return self._explainer
        return self._layer_explainer

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor]],
        targets: Tensor
    ) -> Union[Tensor, Tuple[Tensor]]:
        
        forward_args, additional_forward_args = self._extract_forward_args(inputs)
        attrs = self.explainer.attribute(
            inputs=forward_args,
            target=targets,
            additional_forward_args=additional_forward_args,
        )
        if isinstance(attrs, list):
            attrs = tuple(attrs)
        if isinstance(attrs, tuple) and len(attrs) == 1:
            attrs = attrs[0]
        return attrs


"""
Class for computing Integrated Gradients attributions for a given model and layer.

Args:
    model (torch.nn.Module): The PyTorch model to explain.
    layer (torch.nn.Module or List[str or torch.nn.Module]):
        The layer(s) for which to compute attributions. To target an input layer, a string of argument name
        is available.
    forward_arg_extractor (Callable[[Tuple[Tensor]], Tensor or Tuple[Tensor]] or None, optional):
        A function to extract arguments for each target layer from the tuple of inputs.
        Defaults to None.
    additional_forward_arg_extractor (Callable[[Tuple[Tensor]], Tensor or Tuple[Tensor]] or None, optional):
        A function to extract additional arguments not to be forwarded through the target layer.
        Defaults to None.
    baseline_fn (Callable or List[Callable] or None, optional):
        The function(s) to generate baseline of forward arguments. Must have same length as a tuple of
        forward arguments extracted by `forward_arg_extractor`. Defaults to None.
    n_step (int, optional): The number of steps for numerical approximation. Defaults to 20.

Raises:
    AssertionError: If the type of the `layer` or `baseline_fn` argument is not correct.

Example:
    For a given VQA model and dataset,
    ```
    # ./models.py

    class MyVQAModel(Module):
        ...

        def forward(img, qst, qst_len):
            x = self.vision_model(img)
            embedded = self.embedding(qst)
            y = self.question_model(embedded, qst_len)
            z = self.answer_model(x, y)
            return z
    
    # ./dataset.py
    class MyVQADataset(Dataset):
        ...

        def __getitem__(self, idx):
            ...
            return img, qst, qst_len
                
    ```

    Computes Integrated Gradients attributions for input image and embedded question.

    ```
    # model and data
    model = MyVQAModel().eval()
    dataloader = DataLoader(MyVQADataset(), batch_size=8, shuffle=False)
    imgs, qsts, qst_lens, answers = next(iter(dataloader))
    inputs = (imgs, qsts, qst_lens)
    outputs = model(*inputs)
    targets = outputs.argmax(1)

    # explainer
    layer_ig = LayerIntegratedGradients(
        model=model,
        layer=["img", model.embedding],
        forward_arg_extractor=lambda inputs: tuple(inputs[:2]),     # (imgs, qsts)
        additional_forward_arg_extractor=lambda inputs: inputs[-1], # qst_lens
        baseline_fn=[
            lambda imgs: torch.zeros_like(imgs),        # baseline function for images
            lambda qsts: torch.zeros_like(qsts).long(), # baseline function for questions
        ],
        n_step=20,
    )
    
    # attribute
    img_attrs, qst_attrs = layer_ig.attribute(inputs, targets)
    ```
"""