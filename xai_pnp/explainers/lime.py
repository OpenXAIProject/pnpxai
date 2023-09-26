from typing import Any, List, Callable, Optional

from torch import Tensor
from captum.attr import LimeBase as CaptumLimeBase
from captum.attr import Lime as CaptumLime

from xai_pnp.core._types import Model, DataSource
from xai_pnp.explainers._explainer import Explainer

'''
[NOTE, GH] For LIME explainers, arg `model` for `run` method is replaced to
`forward_func` in order to follow captum's way.

As far as I know, there are more explainers requiring `forward_func` rather
than `model`. This is useful when it is necessary to change a model's forward
function in order to implement explanation: see tutorials/text_classification
(a model with `nn.EmbeddingBag` requires such replacement).
'''

class LimeBase(Explainer):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def run(
            self,
            forward_func: Callable,
            data: DataSource,
            *args: Any,
            **kwargs: Any
        ) -> List[Tensor]:
        lime_base = CaptumLimeBase(forward_func, **self.kwargs)

        attributions = []

        if type(data) is Tensor:
            data = [data]
        
        for datum in data:
            attributions.append(lime_base.attribute(datum, *args, **kwargs))
        
        return attributions


class Lime(Explainer):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(
            self,
            forward_func: Callable,
            data: DataSource,
            *args: Any,
            **kwargs: Any
        ) -> List[Tensor]:
        lime = CaptumLime(forward_func, **self.kwargs)

        attributions = []

        if type(data) is Tensor:
            data = [data]
        
        for datum in data:
            attributions.append(lime.attribute(datum, *args, **kwargs))
        
        return attributions
