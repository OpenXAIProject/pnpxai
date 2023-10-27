from zennit.attribution import (
	Gradient,
	IntegratedGradients,
	SmoothGrad,
)

SUPPORTED_ATTRIBUTORS = {
	'gradient': Gradient,
	'integrated_gradients': IntegratedGradients,
	'smooth_grad': SmoothGrad,
}
