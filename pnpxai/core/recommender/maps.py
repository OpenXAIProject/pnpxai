from pnpxai.explainers import *
from pnpxai.metrics import *
    

DEFAULT_EXPLAINER_MAP_VALUES = [
    (GradCam, ["image"], ["cnn"]),
    (GuidedGradCam, ["image"], ["cnn"]),
    (Lime, ["image", "tabular"], ["linear", "cnn", "rnn", "transformer"]),
    (KernelShap, ["image", "tabular"], ["linear", "cnn", "rnn", "transformer"]),
    (IntegratedGradients, ["image", "text"], ["linear", "cnn", "rnn", "transformer"]),
    (LRPUniformEpsilon, ["image", "text"], ["linear", "cnn", "rnn", "transformer"]),
    (LRPEpsilonGammaBox, ["image", "text"], ["cnn"]),
    (LRPEpsilonPlus, ["image", "text"], ["cnn"]),
    (LRPAlpha2Beta1, ["image", "text"], ["cnn"]),
]


TASK_TO_EXPLAINERS = {
    "image": {
        GradCam,
        GuidedGradCam,
        Lime,
        KernelShap,
        IntegratedGradients,
        LRPUniformEpsilon,
        LRPEpsilonGammaBox,
        LRPEpsilonPlus,
        LRPAlpha2Beta1,
        # RAP,
    },
    "tabular": {
        Lime,
        KernelShap,
        # PDP,
        # CEM,
        # Anchors,
    },
    "text": {
        IntegratedGradients,
        LRPUniformEpsilon,
        LRPEpsilonGammaBox,
        LRPEpsilonPlus,
        LRPAlpha2Beta1,
        # RAP,
        # FullGrad,
        # CEM,
    },
}

ARCHITECTURE_TO_EXPLAINERS = {
    "linear": {
        Lime,
        KernelShap,
        IntegratedGradients,
        LRPUniformEpsilon,
        LRPEpsilonGammabox,
        LRPEpsilonPlus,
        LRPAlpha2Beta1,
        # RAP,
        # FullGrad,
        # CEM,
        # TCAV,
        # Anchors
    },
    "cnn": {
        GradCam,
        GuidedGradCam,
        Lime,
        KernelShap,
        IntegratedGradients,
        LRPUniformEpsilon,
        LRPEpsilonGammabox,
        LRPEpsilonPlus,
        LRPAlpha2Beta1,
        # FullGrad,
        # RAP,
        # CEM,
        # TCAV,
        # Anchors,
    },
    "rnn": {
        Lime,
        KernelShap,
        IntegratedGradients,
        LRPUniformEpsilon,
        LRPEpsilonGammabox,
        LRPEpsilonPlus,
        LRPAlpha2Beta1,
        # RAP,
        # FullGrad,
        # CEM,
        # TCAV,
        # Anchors,
    },
    "transformer": {
        Lime,
        KernelShap,
        LRPUniformEpsilon,
        IntegratedGradients,
        # FullGrad,
        # CEM,
        # TCAV,
        # Anchors
    },
}

EXPLAINER_TO_METRICS = {
    # Correctness -- MuFidelity, Conitinuity -- Sensitivity, Compactness -- Complexity
    # GradCam: {MuFidelity, Sensitivity, Complexity},
    GuidedGradCam: {MuFidelity, Sensitivity, Complexity},
    Lime: {MuFidelity, Sensitivity, Complexity},
    KernelShap: {MuFidelity, Sensitivity, Complexity},
    IntegratedGradients: {MuFidelity, Sensitivity, Complexity},
    LRPUniformEpsilon: {MuFidelity, Sensitivity, Complexity},
    LRPEpsilonGammabox: {MuFidelity, Sensitivity, Complexity},
    LRPEpsilonPlus: {MuFidelity, Sensitivity, Complexity},
    LRPAlpha2Beta1: {MuFidelity, Sensitivity, Complexity},
    # FullGrad: {MuFidelity, Sensitivity, Complexity},
    # RAP: {MuFidelity, Sensitivity, Complexity},

    # # Evaluation metric not implemented yet
    # PDP: {},
    # CEM: {MuFidelity, Sensitivity},
    # TCAV: {MuFidelity, Sensitivity},
    # Anchors: {MuFidelity, Sensitivity},
}
