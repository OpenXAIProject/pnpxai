from pnpxai.explainers._explainer import Explainer, ExplainerWArgs
from pnpxai.explainers.guided_grad_cam import GuidedGradCam
from pnpxai.explainers.integrated_gradients import IntegratedGradients
from pnpxai.explainers.kernel_shap import KernelShap
from pnpxai.explainers.lime import Lime
from pnpxai.explainers.rap import RAP
# TODO: Implement custom LRP
from pnpxai.explainers.lrp import LRP

# TODO: Implement these explainers
from pnpxai.explainers.anchors import Anchors
from pnpxai.explainers.cem import CEM
from pnpxai.explainers.full_grad import FullGrad
from pnpxai.explainers.pdp import PDP
from pnpxai.explainers.tcav import TCAV

AVAILABLE_EXPLAINERS = [
    GuidedGradCam, Lime, KernelShap, IntegratedGradients, FullGrad, LRP, RAP, CEM, TCAV, Anchors, PDP
]
