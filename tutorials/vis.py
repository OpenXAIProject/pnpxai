from functools import partial

from torch.utils.data import DataLoader
import plotly.express as px

from pnpxai.core.project import Project

from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image

model, transform = get_torchvision_model("resnet18")
dataset = get_imagenet_dataset(transform, subset_size=8)
loader = DataLoader(dataset, batch_size=8)
inputs, labels = next(iter(loader))
targets = labels

proj = Project("test proj")
expr = proj.create_experiment(model, auto=True)
expr.run(inputs, targets)


def target_to_label(target):
    return dataset.dataset.idx_to_labels[str(target.item())]

data = expr.runs[0].to_heatmap_data(
    input_process=partial(denormalize_image, mean=transform.mean, std=transform.std),
    target_process=target_to_label,
)

fig = px.imshow(data[0].explanation)
fig.show()