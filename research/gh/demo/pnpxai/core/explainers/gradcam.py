import copy
from typing import Optional

from torch import Tensor
from torch.nn import Module
from torchcam.methods.gradient import _GradCAM
from torchvision.transforms import Resize, InterpolationMode

from .base import ExplainerInterface
from ._utils.cam.extractors import SUPPORTED_GRADCAM_EXTRACTORS


class GradCAMBase(ExplainerInterface):
	def __init__(
		self,
		model: Module,
		extractor_type: str,
		target_layer: Optional[str] = None,
		interpolation_mode: Optional[str] = None,
	):
		super().__init__(
			model = model,
		)
		self.extractor_type = extractor_type
		self.target_layer = target_layer
		self._set_interpolation_mode(interpolation_mode)
	
	def attribute(
		self,
		inputs: Tensor,
		targets: Tensor,
	) -> Tensor:
		_model = copy.deepcopy(self.model)
		extractor = self._get_extractor(_model, inputs.shape[1:])
		outputs = _model(inputs)
		attributions = extractor(
			class_idx = targets.tolist(),
			scores = outputs,
		)[0]
		resizer = self._get_resizer(inputs.shape[2:])
		return resizer(attributions)
	
	def _set_interpolation_mode(self, interpolation_mode):
		self.interpolation_mode = interpolation_mode if interpolation_mode else 'BICUBIC'
	
	def _get_extractor(self, _model, input_shape) -> _GradCAM:
		return SUPPORTED_GRADCAM_EXTRACTORS[self.extractor_type](
			model = _model,
			target_layer = self.target_layer,
			input_shape = input_shape,
		)
	
	def _get_resizer(self, resize_size) -> Resize:
		return Resize(
			size = resize_size,
			interpolation = getattr(InterpolationMode, self.interpolation_mode)
		)


class GradCAM(GradCAMBase):
	def __init__(
		self,
		model: Module,
		target_layer: Optional[str] = None,
		interpolation_mode: Optional[str] = None,
	):
		super().__init__(
			model = model,
			extractor_type = 'gradcam',
			target_layer = target_layer,
			interpolation_mode = interpolation_mode,
		)


class GradCAMpp(GradCAMBase):
	def __init__(
		self,
		model: Module,
		target_layer: Optional[str] = None,
		interpolation_mode: Optional[str] = None,
	):
		super().__init__(
			model = model,
			extractor_type = 'gradcam_pp',
			target_layer = target_layer,
			interpolation_mode = interpolation_mode,
		)


class SmoothGradCAMpp(GradCAMBase):
	def __init__(
		self,
		model: Module,
		target_layer: Optional[str] = None,
		interpolation_mode: Optional[str] = None,
	):
		super().__init__(
			model = model,
			extractor_type = 'smooth_gradcam_pp',
			target_layer = target_layer,
			interpolation_mode = interpolation_mode,
		)


class XGradCAM(GradCAMBase):
	def __init__(
		self,
		model: Module,
		target_layer: Optional[str] = None,
		interpolation_mode: Optional[str] = None,
	):
		super().__init__(
			model = model,
			extractor_type = 'x_gradcam',
			target_layer = target_layer,
			interpolation_mode = interpolation_mode,
		)


class LayerCAM(GradCAMBase):
	def __init__(
		self,
		model: Module,
		target_layer: Optional[str] = None,
		interpolation_mode: Optional[str] = None,
	):
		super().__init__(
			model = model,
			extractor_type = 'layer_cam',
			target_layer = target_layer,
			interpolation_mode = interpolation_mode,
		)



SUPPORTED_GRADCAM = {
	'gradcam': GradCAM,
	'gradcam_pp': GradCAMpp,
	'smooth_gradcam_pp': SmoothGradCAMpp,
	'x_gradcam': XGradCAM,
	'layer_cam': LayerCAM,
}





















