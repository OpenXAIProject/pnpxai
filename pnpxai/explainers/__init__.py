from .grad_cam import GradCam
from .guided_grad_cam import GuidedGradCam
from .gradient import Gradient
from .grad_x_input import GradientXInput
from .smooth_grad import SmoothGrad
from .var_grad import VarGrad
from .integrated_gradients import IntegratedGradients
from .lrp import (
    LRPBase,
    LRPUniformEpsilon,
    LRPEpsilonGammaBox,
    LRPEpsilonPlus,
    LRPEpsilonAlpha2Beta1,
)
from .kernel_shap import KernelShap
from .lime import Lime

from .attention_rollout import (
    AttentionRollout,
    TransformerAttribution,
)


CAM_BASED_EXPLAINERS = [GradCam, GuidedGradCam]
GRADIENT_BASED_EXPLAINERS = [
    Gradient,
    GradientXInput,
    SmoothGrad,
    VarGrad,
    IntegratedGradients,
    LRPUniformEpsilon,
    LRPEpsilonPlus,
    LRPEpsilonGammaBox,
    LRPEpsilonAlpha2Beta1
]
PERTURBATION_BASED_EXPLAINERS = [
    KernelShap,
    Lime,
]
ATTENTION_SPECIFIC_EXPLAINERS = [
    AttentionRollout,
    TransformerAttribution,
]
AVAILABLE_EXPLAINERS = [
    GradCam,
    GuidedGradCam,
    Gradient,
    GradientXInput,
    SmoothGrad,
    VarGrad,
    IntegratedGradients,
    LRPUniformEpsilon,
    LRPEpsilonPlus,
    LRPEpsilonGammaBox,
    LRPEpsilonAlpha2Beta1,
    KernelShap,
    Lime,
    AttentionRollout,
    TransformerAttribution,
]