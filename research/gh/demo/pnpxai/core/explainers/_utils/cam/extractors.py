from torchcam.methods.gradient import (
	GradCAM,
	GradCAMpp,
	SmoothGradCAMpp,
	XGradCAM,
	LayerCAM,
)

SUPPORTED_GRADCAM_EXTRACTORS = {
	'gradcam': GradCAM,
	'gradcam_pp': GradCAMpp,
	'smooth_gradcam_pp': SmoothGradCAMpp,
	'x_gradcam': XGradCAM,
	'layer_cam': LayerCAM,
}
