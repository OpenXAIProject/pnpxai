import torch
from torch.utils.data import DataLoader

# Load Plug and Play XAI Manager
from pnpxai.utils import set_seed
from pnpxai import Project

from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image


set_seed(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def input_visualizer(x): return denormalize_image(
    x, transform.mean, transform.std)


def target_visualizer(x): return dataset.dataset.idx_to_label(x.item())


project = Project('Test Project 1')

model, transform = get_torchvision_model("vit_h_14")
model = model.to(device)
dataset = get_imagenet_dataset(transform, subset_size=25)
loader = DataLoader(dataset, batch_size=10)
def input_extractor(x): return x[0].to(device)
def target_extractor(x): return x[1].to(device)


experiment_resnet = project.create_auto_experiment(
    model,
    loader,
    name='Resnet Experiment',
    input_extractor=input_extractor,
    target_extractor=target_extractor,
    input_visualizer=input_visualizer,
    target_visualizer=target_visualizer,
)
project.get_server().serve(debug=True, host='0.0.0.0', port=5001)
