import torch
from torch.utils.data import DataLoader
from pnpxai.evaluator.metrics import (
    MuFidelity,
    AbPC,
    Sensitivity,
)
from helpers import get_imagenet_dataset, get_torchvision_model


# load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, transform = get_torchvision_model('vit_b_16')
model.to(device)

dataset = get_imagenet_dataset(transform=transform)
loader = DataLoader(dataset, shuffle=False, batch_size=1)
batch = next(iter(loader))
inputs, targets = (tnsr.to(device) for tnsr in batch)

# inference
outputs = model(inputs)
preds = outputs.argmax(-1)


# dummy explain function
def dummy_explain_func(inputs, targets):
    return torch.randn_like(inputs).to(inputs.device)


# explain
attrs = dummy_explain_func(inputs, preds)


# evaluate
metrics = {}


for metric_type in [MuFidelity, AbPC, Sensitivity]:
    metric = metric_type(model=model, explainer=dummy_explain_func)
    metrics[metric_type.__name__] = metric.evaluate(inputs, preds, attrs)

print(metrics)