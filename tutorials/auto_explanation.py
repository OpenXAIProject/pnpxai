import torch
from torch.utils.data import DataLoader

# Load Plug and Play XAI Manager
from pnpxai.utils import set_seed
from pnpxai import Project

from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

set_seed(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def input_visualizer(x): return denormalize_image(x, transform.mean, transform.std)
def target_visualizer(x): return dataset.dataset.idx_to_label(x.item())

# -----------------------------------------------------------------------------#
# ----------------------------------- Project -----------------------------------#
# -----------------------------------------------------------------------------#

project = Project('Test Project 1')

# -----------------------------------------------------------------------------#
# ------------------------------------ Model1 -----------------------------------#
# -----------------------------------------------------------------------------#

model, transform = get_torchvision_model("resnet18")
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

# -----------------------------------------------------------------------------#
# ---------------------------------- Model2 ----------------------------------#
# -----------------------------------------------------------------------------#

model, transform = get_torchvision_model("vit_b_16")
model = model.to(device)
dataset = get_imagenet_dataset(transform, subset_size=25)
loader = DataLoader(dataset, batch_size=10)
def input_extractor(x): return x[0].to(device)
def target_extractor(x): return x[1].to(device)


experiment_vit = project.create_auto_experiment(
    model,
    loader,
    name='ViT Experiment',
    input_extractor=input_extractor,
    target_extractor=target_extractor,
    input_visualizer=input_visualizer,
    target_visualizer=target_visualizer,
)


# Add Project 2 For Testing
project2 = Project('Test Project 2')
model, transform = get_torchvision_model("vit_b_16")
model = model.to(device)

experiment_project2 = project2.create_auto_experiment(
    model,
    loader,
    name='ViT Experiment for Project 2',
    input_extractor=input_extractor,
    target_extractor=target_extractor,
    input_visualizer=input_visualizer,
    target_visualizer=target_visualizer,
)


# Running all project by running one of the projects#
project.get_server().serve(debug=True, host='0.0.0.0', port=5001)
