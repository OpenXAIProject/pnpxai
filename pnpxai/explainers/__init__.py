from pnpxai.explainers._explainer import Explainer, ExplainerWArgs
from pnpxai.explainers.grad_cam import GradCam
from pnpxai.explainers.guided_grad_cam import GuidedGradCam
from pnpxai.explainers.integrated_gradients import IntegratedGradients
from pnpxai.explainers.kernel_shap import KernelShap
from pnpxai.explainers.lime import Lime
from pnpxai.explainers.rap import RAP
from pnpxai.explainers.lrp import LRP
from pnpxai.explainers.deep_lift import DeepLift
from pnpxai.explainers.ts_mule import TSMule

# TODO: Implement these explainers
from pnpxai.explainers.anchors import Anchors
from pnpxai.explainers.cem import CEM
from pnpxai.explainers.full_grad import FullGrad
from pnpxai.explainers.pdp import PDP
from pnpxai.explainers.tcav import TCAV

from typing import List, Type

AVAILABLE_EXPLAINERS: List[Type[Explainer]] = [
    Lime, KernelShap, GuidedGradCam, GradCam, IntegratedGradients, LRP, RAP, DeepLift, TSMule
]
