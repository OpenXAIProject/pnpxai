from typing import Tuple
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.grad_cam import GradCam
from pnpxai.explainers.guided_grad_cam import GuidedGradCam
from pnpxai.explainers.gradient import Gradient
from pnpxai.explainers.grad_x_input import GradientXInput
from pnpxai.explainers.smooth_grad import SmoothGrad
from pnpxai.explainers.var_grad import VarGrad
from pnpxai.explainers.integrated_gradients import IntegratedGradients
from pnpxai.explainers.lrp import (
    LRPBase,
    LRPUniformEpsilon,
    LRPEpsilonGammaBox,
    LRPEpsilonPlus,
    LRPEpsilonAlpha2Beta1,
)
from pnpxai.explainers.rap import RAP
from pnpxai.explainers.kernel_shap import KernelShap
from pnpxai.explainers.deep_lift_shap import DeepLiftShap
from pnpxai.explainers.lime import Lime
from pnpxai.explainers.attention_rollout import (
    AttentionRollout,
    TransformerAttribution,
)
from pnpxai.explainers.lear import LEAR
from pnpxai.explainers.gfgp import Gfgp

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
    Gfgp
]
ATTENTION_SPECIFIC_EXPLAINERS = [
    AttentionRollout,
    TransformerAttribution,
]
AVAILABLE_EXPLAINERS: Tuple[Explainer] = (
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
    RAP,
    KernelShap,
    Lime,
    Gfgp,
    AttentionRollout,
    TransformerAttribution,
)

EXPLAINERS_FOR_TABULAR = []
EXPLAINERS_FOR_MEDICAL_IMAGE: Tuple[Explainer] = (LEAR,)