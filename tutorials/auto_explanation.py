import os
import torch
from torch.utils.data import DataLoader

# Load Plug and Play XAI Manager
from pnpxai.utils import set_seed
from pnpxai import Project

from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

# Set the seed for reproducibility
set_seed(seed=0)

# Determine the device based on the availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------#
# --------------------------------- project 1 ---------------------------------#
# -----------------------------------------------------------------------------#

# Initialize Project 1
project = Project('Test Project 1', config='config.example.yml')


# -----------------------------------------------------------------------------#
# ------------------------------- experiment 1-1 ------------------------------#
# -----------------------------------------------------------------------------#

# Load the model and its pre-processing transform for experiment 1-1
model, transform = get_torchvision_model("resnet18")
model = model.to(device)

# Prepare the dataset and dataloader for experiment 1-1
data_root_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data/ImageNet')
dataset = get_imagenet_dataset(
    transform, subset_size=25, root_dir=data_root_dir)
loader = DataLoader(dataset, batch_size=10)

# Define functions to extract input and target from the data loader


def input_extractor(x): return x[0].to(device)
def target_extractor(x): return x[1].to(device)

# Define helper functions to visualize input and target


def input_visualizer(x): return denormalize_image(
    x, transform.mean, transform.std)


def target_visualizer(x): return dataset.dataset.idx_to_label(x.item())


# Create an experiment to explain the defined model and dataset
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
# ------------------------------- experiment 1-2 ------------------------------#
# -----------------------------------------------------------------------------#

# Load the model and its pre-processing transform for experiment 1-2
model, transform = get_torchvision_model("vit_b_16")
model = model.to(device)

# Prepare the dataset and dataloader for experiment 1-2
dataset = get_imagenet_dataset(
    transform, subset_size=25, root_dir=data_root_dir)
loader = DataLoader(dataset, batch_size=10)

# Define functions to extract input and target from the data loader


def input_extractor(x): return x[0].to(device)
def target_extractor(x): return x[1].to(device)

# Define helper functions to visualize input and target


def input_visualizer(x): return denormalize_image(
    x, transform.mean, transform.std)


def target_visualizer(x): return dataset.dataset.idx_to_label(x.item())


# Create an experiment to explain the defined model and dataset
# Use predefined config, since it is not AutoExperiment
experiment_vit = project.create_experiment(
    model,
    loader,
    name='ViT Experiment',
    input_extractor=input_extractor,
    target_extractor=target_extractor,
    input_visualizer=input_visualizer,
    target_visualizer=target_visualizer,
)


# -----------------------------------------------------------------------------#
# --------------------------------- project 2 ---------------------------------#
# -----------------------------------------------------------------------------#

# Initialize Project 2
project2 = Project('Test Project 2')


# -----------------------------------------------------------------------------#
# ------------------------------- experiment 2-1 ------------------------------#
# -----------------------------------------------------------------------------#

# Load the model and its pre-processing transform for experiment 2-1
model, transform = get_torchvision_model("vit_b_16")
model = model.to(device)

# Prepare the dataset and dataloader for experiment 2-1
dataset = get_imagenet_dataset(transform, subset_size=25)
loader = DataLoader(dataset, batch_size=10)

# Define functions to extract input and target from the data loader


def input_extractor(x): return x[0].to(device)
def target_extractor(x): return x[1].to(device)

# Define helper functions to visualize input and target


def input_visualizer(x): return denormalize_image(
    x, transform.mean, transform.std)


def target_visualizer(x): return dataset.dataset.idx_to_label(x.item())


# Create an experiment to explain the defined model and dataset
experiment_project2 = project2.create_auto_experiment(
    model,
    loader,
    name='ViT Experiment for Project 2',
    input_extractor=input_extractor,
    target_extractor=target_extractor,
    input_visualizer=input_visualizer,
    target_visualizer=target_visualizer,
)


# -----------------------------------------------------------------------------#
# ----------------------------------- launch ----------------------------------#
# -----------------------------------------------------------------------------#

# Launch the interactive web-based dashboard by running one of the projects defined above
project.get_server().serve(host='0.0.0.0', port=5000)
