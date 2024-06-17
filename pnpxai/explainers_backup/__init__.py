from pnpxai.explainers_backup._explainer import Explainer, ExplainerWArgs
from pnpxai.explainers_backup.grad_cam import GradCam
from pnpxai.explainers_backup.guided_grad_cam import GuidedGradCam
from pnpxai.explainers_backup.integrated_gradients import IntegratedGradients
from pnpxai.explainers_backup.kernel_shap import KernelShap
from pnpxai.explainers_backup.lime import Lime
from pnpxai.explainers_backup.rap import RAP
from pnpxai.explainers_backup.lrp import LRP
from pnpxai.explainers_backup.deep_lift import DeepLift
from pnpxai.explainers_backup.ts_mule import TSMule

# TODO: Implement these explainers_backup
from pnpxai.explainers_backup.anchors import Anchors
from pnpxai.explainers_backup.cem import CEM
from pnpxai.explainers_backup.full_grad import FullGrad
from pnpxai.explainers_backup.pdp import PDP
from pnpxai.explainers_backup.tcav import TCAV

from typing import List, Type

AVAILABLE_EXPLAINERS: List[Type[Explainer]] = [
    Lime, KernelShap, GuidedGradCam, GradCam, IntegratedGradients, LRP, RAP, DeepLift, TSMule
]
