from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gradio as gr

import torch
from torch.utils.data import DataLoader
import torchvision

from pnpxai.explainers import IntegratedGradients, ExplainerWArgs, LRP, RAP

from helpers import get_imagenet_dataset, get_torchvision_model
from helpers import get_livertumor_dataset, get_livertumor_model
from pnpxai.explainers.utils.post_process import postprocess_attr

import pdb

device = 'cuda'
batch_size = 4 # 32

def postprocess_attr(attr):
    # l1-norm
    # attr = torch.abs(attr)
    # summed_tensor = attr.sum(dim=1)
    
    # l2-norm-sq
    # attr = attr ** 2
    # summed_tensor = attr.sum(dim=1)

    # sum, pos
    summed_tensor = attr.sum(dim=1)
    summed_tensor = torch.nn.functional.relu(summed_tensor)

    max_values = summed_tensor.view(summed_tensor.shape[0], -1).max(dim=1)[0].unsqueeze(1).unsqueeze(2)
    min_values = summed_tensor.view(summed_tensor.shape[0], -1).min(dim=1)[0].unsqueeze(1).unsqueeze(2)
    postprocessed = (summed_tensor - min_values) / (max_values - min_values)
    return postprocessed.cpu().detach().numpy()

sample_to_np = lambda sample: sample.permute(1, 2, 0).squeeze(dim=-1).detach().numpy()
def denormalize_sample(inputs, mean, std):
    return sample_to_np(
        inputs
        * torch.tensor(std)
        + torch.tensor(mean)
    )

def idx_to_label(idx):
    if idx == 0:
        return 'Normal'
    elif idx == 1:
        return 'Tumor'

model, transform = get_livertumor_model('../models/best_model.pt')
model.to(device)
dataset = get_livertumor_dataset(subset_size=128, transform=transform)

explainers = OrderedDict([
    ('IntegratedGradients', IntegratedGradients(model)),
    ('LRP', LRP(model)),
    ('RAP', RAP(model))
])

loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=2)
for batch_idx, (inputs, labels, masks) in enumerate(loader):
    inputs = inputs.to(device)
    targets = labels
    with torch.no_grad():
        preds = model(inputs).argmax(dim=1)
    masks = torchvision.transforms.Resize((224, 224), antialias=True)(masks)

    # Compute input attributions
    attrs = dict()
    # pdb.set_trace()
    for key, explainer in explainers.items():
    # with torch.autograd.enable_grad():
        attrs[key] = explainer.attribute(inputs, targets, n_classes=2)

    # Postprocess input attributions
    for key, attr in attrs.items():
        attrs[key] = postprocess_attr(attr)

    '''
    num_samples = len(inputs)
    num_expls = len(explainers)
    for s in range(num_samples):
        ncols = 2 + num_expls
        fig, axs = plt.subplots(1, ncols, figsize=(5*ncols, 5))
        axs[0].imshow(denormalize_sample(inputs[s].cpu(), [.5], [.5]), cmap='gray')
        label = idx_to_label(labels[s].item())
        pred = idx_to_label(preds[s].item())
        axs[0].set_title(f'Lable: {label} / Pred: {pred}')

        axs[1].imshow(masks[s], cmap='gray', norm=mcolors.Normalize(vmin=0, vmax=2))
        axs[1].set_title('GT Seg. Mask')

        for e, (key, attr) in enumerate(attrs.items()):
            axs[2+e].imshow(attr[s], cmap='coolwarm')
            axs[2+e].set_title(key)

        [ax.axis('off') for ax in axs]
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'results_liver_tumor/{batch_idx * batch_size + s}')
        plt.close(fig)
    '''
    def visualize(inputs, labels, masks, attrs, preds):
        num_samples = len(inputs)
        num_expls = len(explainers)
        outputs = []
        for s in range(num_samples):
            output = []
            output.append(denormalize_sample(inputs[s].cpu(), [.5], [.5]))
            label = idx_to_label(labels[s].item())
            pred = idx_to_label(preds[s].item())
            output.append((f'Lable: {label} / Pred: {pred}',))

            output.append(masks[s])
            output.append(('GT Seg. Mask',))

            for e, (key, attr) in enumerate(attrs.items()):
                output.append(attr[s])
                output.append((key,))

            outputs.append(output)
        return outputs

    outputs = visualize(inputs, labels, masks, attrs, preds)
    gr.Interface(fn=visualize, inputs=gr.Image(), outputs='image').launch(share=True)
    pdb.set_trace()