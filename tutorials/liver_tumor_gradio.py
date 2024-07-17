from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision

from pnpxai.explainers import IntegratedGradients, LRP, RAP, KernelShap, Lime

from helpers import get_imagenet_dataset, get_torchvision_model
from helpers import get_livertumor_dataset, get_livertumor_model
from pnpxai.explainers.utils.post_process import postprocess_attr

import gradio as gr

import pdb

device = 'cuda'
batch_size = 32

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
dataset = get_livertumor_dataset(subset_size=128, transform=transform)
model.to(device)

explainers = OrderedDict([
    ('IntegratedGradients', IntegratedGradients(model)),
    ('LRP', LRP(model)),
    ('RAP', RAP(model)),
    ('KernelSHAP', KernelShap(model)),
    ('Lime', Lime(model)),
])

def load_samples(dataset, num_samples=20):
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
    samples = next(iter(dataloader))
    return [(inputs, labels, masks) for inputs, labels, masks in zip(*samples)]

def get_examples(samples):
    examples = []
    for inputs, labels, masks in samples:
        original_image = denormalize_sample(inputs.cpu(), [.5], [.5])
        examples.append((original_image, labels.item(), masks))
    return examples

def explain(index):
    inputs, labels, masks = dataset[int(index)]

    # Preprocess
    inputs = inputs.to(device)[None, ...]
    targets = labels[None, ...]
    masks = torch.tensor(masks)[None, ...]
    masks = torchvision.transforms.Resize((224, 224), antialias=True)(masks)
    with torch.no_grad():
        preds = model(inputs).argmax(dim=1)

    # Compute input attributions
    attrs = dict()
    for key, explainer in explainers.items():
        try:
            attrs[key] = explainer.attribute(inputs, targets, n_classes=2)
        except:
            attrs[key] = explainer.attribute(inputs, targets)

    # Postprocess input attributions
    for key, attr in attrs.items():
        attrs[key] = postprocess_attr(attr)

    # Prepare data for Gradio display
    cmap = plt.get_cmap('coolwarm')
    original_image = denormalize_sample(inputs.squeeze(dim=0).cpu(), [.5], [.5])
    gt_mask = masks.squeeze(dim=0).cpu().numpy()
    gt_mask = gt_mask / 2.
    # explanation_images = {key: attr for key, attr in attrs.items()}
    explanation_images = [cmap(attr[0]) for _, attr in attrs.items()]
    # explanation_images = explanation_images['IntegratedGradients'][0]
    label = idx_to_label(labels.item())
    pred = idx_to_label(preds.item())

    results = (label, pred, original_image, gt_mask) + tuple(explanation_images)

    # pdb.set_trace()

    # return label, pred, original_image, gt_mask, explanation_images
    return results

# Gradio interface definition
samples = load_samples(dataset, num_samples=20)
examples = get_examples(samples)
# pdb.set_trace()

outputs = [
    gr.Text(label="Label"),
    gr.Text(label="Prediction"),
    gr.Image(type="numpy", label="Original Image"),
    gr.Image(type="numpy", label="Ground Truth Mask"),
    # gr.Image(type="numpy", label="Explanation"),
]
outputs.extend([gr.Image(type="numpy", label=key) for key in explainers.keys()])

interface = gr.Interface(
    fn=explain,
    # fn=explain_wrapper,
    # inputs=[
    #     gr.Image(type="numpy", label="Input Image"),
    #     gr.Textbox(type="text", label="Label"),
    #     gr.Image(type="numpy", label="Ground Truth Mask"),
    # ],
    inputs=gr.Textbox(type="text", label="Index"),
    # outputs=[
    #     gr.Text(label="Label"),
    #     gr.Text(label="Prediction"),
    #     gr.Image(type="numpy", label="Original Image"),
    #     gr.Image(type="numpy", label="Ground Truth Mask"),
    #     gr.Image(type="numpy", label="Explanation"),
    # ],
    outputs=outputs,
    description=
    "**Label** | **Prediction** | **Input Image** | **Ground Truth Mask** | **Explanation**",
    css="""
            .output-image {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-around;
            }
    """,
    # examples=examples,
)

# Launch the Gradio interface
interface.launch(share=True)
