from typing import Literal, Union, Sequence

import torch
import numpy as np
from skimage.segmentation import (
    felzenszwalb,
    quickshift,
    slic,
    watershed,
)

from pnpxai.explainers.base import Tunable
from pnpxai.explainers.types import TunableParameter
from pnpxai.explainers.utils.base import UtilFunction
from torchvision.transforms import InterpolationMode, Resize
import torchvision.transforms.functional as TF


class FeatureMaskFunction(UtilFunction):
    """
    A base class for feature mask functions used in attribution methods. Feature mask functions 
    generate masks for input features, which are useful in various attribution methods to 
    highlight or hide certain parts of the input data.

    Methods:
        __init__():
            Initializes the `FeatureMaskFunction` instance. This constructor is provided for 
            completeness and can be overridden by subclasses if needed.

        _skseg_for_tensor(fn, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
            Applies a segmentation function `fn` to each tensor in `inputs` to generate feature masks. 
            The function converts the tensor to a NumPy array, applies the segmentation function, 
            and then converts the result back to a tensor. This method is useful for generating masks 
            for image or spatial data.

    Notes:
        - `FeatureMaskFunction` is designed to be subclassed. Concrete feature mask functions 
          should inherit from this class and implement the actual masking logic.
        - The `_skseg_for_tensor` method is a static method that handles the process of applying 
          a segmentation function to tensors. This method is intended to be used within subclasses 
          that need to generate feature masks.
        - Ensure that the function `fn` provided is compatible with the shape and type of the input 
          tensors.
    """
    
    def __init__(self):
        pass

    @staticmethod
    def _skseg_for_tensor(fn, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Applies a segmentation function to each tensor in the input batch to generate feature masks.

        Args:
            fn (Callable): The segmentation function to apply. This function should accept a 
                NumPy array and return a mask in some form.
            inputs (torch.Tensor): A batch of input tensors to process. The tensors are assumed 
                to have dimensions suitable for segmentation.
            **kwargs: Additional keyword arguments passed to the segmentation function.

        Returns:
            torch.Tensor: A tensor of generated feature masks, stacked along the batch dimension. 
                The result is converted to a long tensor and moved to the same device as the input.
        """
        feature_mask = [
            torch.tensor(fn(
                inp.permute(1, 2, 0).detach().cpu().numpy(),
                **kwargs
            )) for inp in inputs
        ]
        return torch.stack(feature_mask).long().to(inputs.device)


class Checkerboard(FeatureMaskFunction):
    def __init__(
        self,
        size: Sequence[int] = [20, 20],
    ):
        assert len(size) == 2
        self.size = size
        self._n_checkers = size[0] * size[1]

    def __call__(self, inputs: torch.Tensor):
        assert inputs.dim() == 4

        bsz, c, h, w = inputs.size()
        # print(input_size)

        resize = Resize([h, w], interpolation=InterpolationMode.NEAREST)

        patch_masks = []
        for i in range(self._n_checkers):
            mask = np.zeros(self._n_checkers)
            mask[i] = i
            mask = resize(torch.Tensor(mask).reshape(-1,self.size[0], self.size[1])).unsqueeze(1)
            patch_masks.append(mask.numpy())
        return torch.from_numpy(sum(patch_masks)).squeeze(1).repeat(bsz, 1, 1).long().to(inputs.device)


class Felzenszwalb(FeatureMaskFunction, Tunable):
    def __init__(
        self,
        scale: float = 250.,
        sigma: float = 1.,
    ):
        FeatureMaskFunction.__init__(self)
        self.scale = TunableParameter(
            name='scale',
            current_value=scale,
            dtype=float,
            is_leaf=True,
            space={'low': 1e0, 'high': 1e3, 'log': True},
        )
        self.sigma = TunableParameter(
            name='sigma',
            current_value=sigma,
            dtype=float,
            is_leaf=True,
            space={'low': 0., 'high': 2., 'step': .05},
        )
        Tunable.__init__(self)
        self.register_tunable_params([self.scale, self.sigma])

    def __call__(self, inputs: torch.Tensor):
        return self._skseg_for_tensor(
            felzenszwalb,
            inputs,
            scale=self.scale.current_value,
            sigma=self.sigma.current_value,
            min_size=50,
        )


class Quickshift(FeatureMaskFunction, Tunable):
    def __init__(
        self,
        ratio: float = 1.,
        kernel_size: float = 5,
        max_dist: int = 10.,
        sigma: float = 0.,
    ):
        FeatureMaskFunction.__init__(self)
        self.ratio = TunableParameter(
            name='ratio',
            current_value=ratio,
            dtype=float,
            is_leaf=True,
            space={'low': 0., 'high': 1., 'step': .1},
        )
        self.kernel_size = TunableParameter(
            name='kernel_size',
            current_value=kernel_size,
            dtype=float,
            is_leaf=True,
            space={'low': 1., 'high': 10., 'step': 1.},
        )
        self.max_dist = TunableParameter(
            name='max_dist',
            current_value=max_dist,
            dtype=int,
            is_leaf=True,
            space={'low': 1, 'high': 20, 'step': 1},
        )
        self.sigma = TunableParameter(
            name='sigma',
            current_value=sigma,
            dtype=float,
            is_leaf=True,
            space={'low': 0., 'high': 2., 'step': .1},
        )
        Tunable.__init__(self)
        self.register_tunable_params([
            self.ratio, self.kernel_size, self.max_dist, self.sigma])

    def __call__(self, inputs: torch.Tensor):
        if inputs.size(1) == 1:
            inputs = inputs.tile(1, 3, 1, 1)
        return self._skseg_for_tensor(
            quickshift,
            inputs,
            ratio=self.ratio.current_value,
            kernel_size=self.kernel_size.current_value,
            max_dist=self.max_dist.current_value,
            sigma=self.sigma.current_value,
        )


class Slic(FeatureMaskFunction, Tunable):
    def __init__(
        self,
        n_segments: int = 150,
        compactness: float = 1.,
        sigma: float = 0.
    ):
        FeatureMaskFunction.__init__(self)
        self.n_segments = TunableParameter(
            name='n_segments',
            current_value=n_segments,
            dtype=int,
            is_leaf=True,
            space={'low': 100, 'high': 500, 'step': 10},
        )
        self.compactness = TunableParameter(
            name='compactness',
            current_value=compactness,
            dtype=float,
            is_leaf=True,
            space={'low': 1e-2, 'high': 1e2, 'log': True},
        )
        self.sigma = TunableParameter(
            name='sigma',
            current_value=sigma,
            dtype=float,
            is_leaf=True,
            space={'low': 0., 'high': 2., 'step': .1},
        )
        Tunable.__init__(self)
        self.register_tunable_params([
            self.n_segments, self.compactness, self.sigma])

    def __call__(self, inputs: torch.Tensor):
        return self._skseg_for_tensor(
            slic,
            inputs,
            n_segments=self.n_segments.current_value,
            compactness=self.compactness.current_value,
            sigma=self.sigma.current_value,
        )


class Watershed(FeatureMaskFunction):
    def __init__(
        self,
        markers: int,
        compactness: float,
    ):
        FeatureMaskFunction.__init__(self)
        self.markers = TunableParameter(
            name='markers',
            current_value=markers,
            dtype=int,
            is_leaf=True,
            space={'low': 10, 'high': 200, 'step': 10},
        )
        self.compactness = TunableParameter(
            name='compactness',
            current_value=compactness,
            dtype=float,
            is_leaf=True,
            space={'low': 1e-6, 'high': 1., 'log': True},
        )
        Tunable.__init__(self)
        self.register_tunable_params([self.markers, self.compactness])

    def __call__(self, inputs: torch.Tensor):
        return self._skseg_for_tensor(
            watershed,
            inputs,
            markers=self.markers.current_value,
            compactness=self.compactness.current_value,
        )


class NoMask1d(FeatureMaskFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        bsz, seq_len = inputs.size()
        seq_masks = torch.arange(seq_len).repeat(bsz).view(bsz, seq_len)
        return seq_masks.to(inputs.device)


class NoMask2d(FeatureMaskFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        bsz, channel, seq_len = inputs.size()
        seq_masks = torch.arange(seq_len).repeat(bsz).view(bsz, channel, seq_len)
        return seq_masks.to(inputs.device)


FEATURE_MASK_FUNCTIONS = {
    (float, 2): {
        'no_mask_1d': NoMask1d,
    },
    (float, 3): {
        'no_mask_2d': NoMask2d,
    },
    (float, 4): {
        'checkerboard': Checkerboard,
        'felzenszwalb': Felzenszwalb,
        'quickshift': Quickshift,
        'slic': Slic,
    },
    (int, 2): {
        'no_mask_1d': NoMask1d,
    }
}
