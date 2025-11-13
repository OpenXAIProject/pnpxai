from typing import Literal, Union

import torch
from skimage.segmentation import (
    felzenszwalb,
    quickshift,
    slic,
    watershed,
)

from pnpxai.explainers.utils.base import UtilFunction


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

    @classmethod
    def from_method(cls, method, **kwargs):
        feature_mask_fn = FEATURE_MASK_FUNCTIONS.get(method, None)
        if feature_mask_fn is None:
            raise ValueError
        return feature_mask_fn(**kwargs)

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
            torch.tensor(fn(inp.permute(1, 2, 0).detach().cpu().numpy(), **kwargs))
            for inp in inputs
        ]
        return torch.stack(feature_mask).long().to(inputs.device)


class Felzenszwalb(FeatureMaskFunction):
    def __init__(
        self,
        scale: float = 250.0,
        sigma: float = 1.0,
        min_size: int = 50,
    ):
        super().__init__()
        self.scale = scale
        self.sigma = sigma

    def __call__(self, inputs: torch.Tensor):
        return self._skseg_for_tensor(
            felzenszwalb,
            inputs,
            scale=self.scale,
            sigma=self.sigma,
            min_size=50,
        )

    def get_tunables(self):
        return {
            "scale": (float, {"low": 1e0, "high": 1e3, "log": True}),
            "sigma": (float, {"low": 0.0, "high": 2.0, "step": 0.05}),
        }


class Quickshift(FeatureMaskFunction):
    def __init__(
        self,
        ratio: float = 1.0,
        kernel_size: float = 5,
        max_dist: float = 10.0,
        sigma: float = 0.0,
    ):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.sigma = sigma

    def __call__(self, inputs: torch.Tensor):
        if inputs.size(1) == 1:
            inputs = inputs.tile(1, 3, 1, 1)
        return self._skseg_for_tensor(
            quickshift,
            inputs,
            ratio=self.ratio,
            kernel_size=self.kernel_size,
            max_dist=self.max_dist,
            sigma=self.sigma,
        )

    def get_tunables(self):
        return {
            "ratio": (float, {"low": 0.0, "high": 1.0, "step": 0.1}),
            "kernel_size": (float, {"low": 1.0, "high": 10.0, "step": 1.0}),
            "max_dist": (int, {"low": 1, "high": 20, "step": 1}),
            "sigma": (
                float,
                {
                    "low": 0.0,
                    "high": 2.0,
                    "step": 0.1,
                },
            ),
        }


class Slic(FeatureMaskFunction):
    def __init__(
        self, n_segments: int = 150, compactness: float = 1.0, sigma: float = 0.0
    ):
        super().__init__()
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma

    def __call__(self, inputs: torch.Tensor):
        return self._skseg_for_tensor(
            slic,
            inputs,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
        )

    def get_tunables(self):
        return {
            "n_segments": (float, {"low": 100, "high": 500, "step": 10}),
            "compactness": (float, {"low": 1e-2, "high": 1e2, "log": True}),
            "sigma": (float, {"low": 0.0, "high": 2.0, "step": 0.1}),
        }


class Watershed(FeatureMaskFunction):
    def __init__(
        self,
        markers: int,
        compactness: float,
    ):
        super().__init__()
        self.markers = markers
        self.compactness = compactness

    def __call__(self, inputs: torch.Tensor):
        return self._skseg_for_tensor(
            watershed,
            inputs,
            markers=self.markers,
            compactness=self.compactness,
        )

    def get_tunables(self):
        return {
            "markers": (int, {"low": 10, "high": 200, "step": 10}),
            "compactness": (float, {"low": 1e-6, "high": 1.0, "log": True}),
        }


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


class NoMaskAdaptive(FeatureMaskFunction):
    def __init__(self):
        super().__init__()

        self._mask_dim_map = {2: NoMask1d(), 3: NoMask2d()}

    def __call__(self, inputs):
        inputs_n_dim = inputs.ndim

        if inputs_n_dim not in self._mask_dim_map:
            raise Exception(
                f"Unsupported number of dimensions ({inputs_n_dim}) ecnountered"
            )

        return self._mask_dim_map[inputs_n_dim](inputs)


FEATURE_MASK_FUNCTIONS_FOR_IMAGE = {
    "felzenszwalb": Felzenszwalb,
    "quickshift": Quickshift,
    "slic": Slic,
    # 'watershed': watershed_for_tensor, TODO: watershed
}

FEATURE_MASK_FUNCTIONS_FOR_TEXT = {
    "no_mask_1d": NoMask1d,
}

FEATURE_MASK_FUNCTIONS_FOR_TIME_SERIES = {
    "no_mask_2d": NoMaskAdaptive,
}

FEATURE_MASK_FUNCTIONS = {
    **FEATURE_MASK_FUNCTIONS_FOR_IMAGE,
    **FEATURE_MASK_FUNCTIONS_FOR_TEXT,
    **FEATURE_MASK_FUNCTIONS_FOR_TIME_SERIES,
}


FeatureMaskMethod = Literal[
    "felzenszwalb", "quickshift", "slic", "watershed", "no_mask_1d"
]
FeatureMaskMethodOrFunction = Union[FeatureMaskMethod, FeatureMaskFunction]
