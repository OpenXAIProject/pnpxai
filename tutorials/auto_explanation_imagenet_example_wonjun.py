#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader
from pnpxai import AutoExplanationForImageClassification, Experiment
from pnpxai.core.modality import ImageModality
from pnpxai.explainers import LRPEpsilonPlus, Gfgp, GradientXInput, Gradient
from pnpxai.explainers.utils.postprocess import Identity, PostProcessor
from pnpxai.evaluator.metrics import MuFidelity, Sensitivity, Complexity, AbPC


from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image

import torch.nn as nn
import torch.nn.functional as F
import torchvision
#%%
# ------------------------------------------------------------------------------#
# -------------------------------- basic usage ---------------------------------#
# ------------------------------------------------------------------------------#

# setup
torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, transform = get_torchvision_model('resnet18')
dataset = get_imagenet_dataset(transform, indices=range(1000))
loader = DataLoader(dataset, batch_size=4, shuffle=False)


explainers = [
    "LRPEpsilonPlus(model=model, epsilon=1e-6, n_classes=1000)",
    "Gradient(model=model)",
    "GradientXInput(model=model)",
    "Gfgp(model=model, transforms=transform, timesteps=[100, 200, 300, 400, 500, 600, 700])",
]
# explainer = LRPEpsilonPlus(model=model, epsilon=1e-6, n_classes=1000)
# explainer = Gradient(model=model)
# explainer = GradientXInput(model=model)
# explainer = Gfgp(model=model, transforms=transform, timesteps=[100, 200, 300, 400, 500, 600, 700])

for _explainer in explainers:
    explainer = eval(_explainer)

    metrics = [
        MuFidelity(model, explainer, n_perturb=10),
        Sensitivity(model, explainer, n_iter=10),
        Complexity(model, explainer)
    ]
    modality = ImageModality()
    expr = Experiment(
        model=model.to(device),
        data=loader,
        modality=modality,
        explainers=[explainer],
        # postprocessors=[Identity()],
        postprocessors=[
            PostProcessor(
                pooling_fn=modality.pooling_fn_selector.select('sumpos'), # ['sumpos', 'sumabs', 'l1norm', 'maxnorm', 'l2norm', 'l2normsq', 'possum', 'posmaxnorm', 'posl2norm', 'posl2normsq']
                normalization_fn=modality.normalization_fn_selector.select('minmax') # ['minmax', 'identity']
            )
        ],
        metrics=metrics,
        input_extractor=lambda x: x[0].to(device),
        label_extractor=lambda x: x[-1].to(device),
        target_extractor=lambda outputs: outputs.argmax(-1).to(device)
    )

    # explain and evaluate
    results = expr.run_batch(
        data_ids=range(5),
        explainer_id=0,
        postprocessor_id=0,
        metric_id=0,
    )


    fig, ax = plt.subplots(1,len(results['inputs']))
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    for i, result in enumerate(results['inputs']):
        result = result.detach().cpu().numpy()
        result = np.transpose(result, (1,2,0))
        result = (result - result.min()) / (result.max() - result.min())
        ax[i].imshow(result)
        ax[i].set_axis_off()
    plt.tight_layout()
    fig.savefig('inputs.png', dpi=300)

    fig, ax = plt.subplots(1,len(results['postprocessed']))
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    for i, result in enumerate(results['postprocessed']):
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        ax[i].imshow(result, cmap='seismic')
        ax[i].set_axis_off()
    plt.tight_layout()
    fig.savefig(_explainer.split('(')[0], dpi=300)
