import torch
from torch.utils.data import DataLoader

# Load Plug and Play XAI Manager
from pnpxai.utils import set_seed
from pnpxai import Project

# from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image
from helpers import get_livertumor_dataset, get_livertumor_model, denormalize_sample

from torchvision import transforms

import pdb



#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

# Set the seed for reproducibility
set_seed(seed=0)

# Determine the device based on the availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#-----------------------------------------------------------------------------#
#--------------------------------- project 1 ---------------------------------#
#-----------------------------------------------------------------------------#

# Initialize Project 1
project = Project('Test Project 1')


#-----------------------------------------------------------------------------#
#------------------------------- experiment 1-1 ------------------------------#
#-----------------------------------------------------------------------------#

# Load the model and its pre-processing transform for experiment 1-1
# model, transform = get_torchvision_model("resnet18")
# model, transform = get_livertumor_model('../models/epoch_09.pt')
model, transform = get_livertumor_model('../models/best_model.pt')
model = model.to(device)

# Prepare the dataset and dataloader for experiment 1-1
dataset = get_livertumor_dataset(
    subset_size=100,
    transform=transform,
)
loader = DataLoader(dataset, batch_size=10)

# Define functions to extract input and target from the data loader
def input_extractor(x): return x[0].to(device).repeat(1, 3, 1, 1)
def target_extractor(x): return x[1].to(device)
# def target_extractor(x): return x[1].to(device).argmax()[None, None]

# Define helper functions to visualize input and target
def input_visualizer(x): return denormalize_sample(x, [.5], [.5])
# def target_visualizer(x): return dataset.dataset.idx_to_label(x.argmax().item())
def target_visualizer(x): print(f'at target_visualizer: {x}'); return dataset.dataset.idx_to_label(x)

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


#-----------------------------------------------------------------------------#
#----------------------------------- launch ----------------------------------#
#-----------------------------------------------------------------------------#

# Launch the interactive web-based dashboard by running one of the projects defined above
project.get_server().serve(debug=True, host='0.0.0.0', port=5002)
