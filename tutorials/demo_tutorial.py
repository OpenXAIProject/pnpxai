import torch
from torch.utils.data import DataLoader
from pnpxai.utils import set_seed
from pnpxai import Project
from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image

# Set seed and device
set_seed(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define common functions
def create_experiment(project, model_name, dataset_subset_size=25, batch_size=10):
    model, transform = get_torchvision_model(model_name)
    model = model.to(device)
    images_for_demo = [0, 86, 148, 163, 596, 608, 840, 983, 916]
    dataset = get_imagenet_dataset(transform, indices=images_for_demo)
    loader = DataLoader(dataset, batch_size=batch_size)
    def input_extractor(x): return x[0].to(device)
    def target_extractor(x): return x[1].to(device)
    def input_visualizer(x): return denormalize_image(x, transform.mean, transform.std)
    def target_visualizer(x): return dataset.dataset.idx_to_label(x.item())
    
    return project.create_auto_experiment(
        model,
        loader, 
        name=f'{model_name} Experiment',
        input_extractor=input_extractor,
        target_extractor=target_extractor,
        input_visualizer=input_visualizer,
        target_visualizer=target_visualizer,
    )

# Create Project 1
project1 = Project('Test Project 1')

# Create experiments for ResNet and ViT
experiment_resnet = create_experiment(project1, "resnet18")
experiment_vit = create_experiment(project1, "vit_b_16")

# Create Project 2
project2 = Project('Test Project 2')

# Create an experiment for ViT in Project 2
experiment_project2 = create_experiment(project2, "vit_b_16")

# Serve projects
project1.get_server().serve(debug=True, host='0.0.0.0', port=5001)

# You can run multiple projects in parallel by running only 1 project
# project2.get_server().serve(debug=True, host='0.0.0.0', port=5002)
