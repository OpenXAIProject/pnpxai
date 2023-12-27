import torch
from torch.utils.data import DataLoader, Subset

from pnpxai.utils import set_seed
from pnpxai import Project

from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

set_seed(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------#
# ----------------------------------- model -----------------------------------#
# -----------------------------------------------------------------------------#
project = Project('test_project')
model, transform = get_torchvision_model("resnet18")
model = model.to(device)

# -----------------------------------------------------------------------------#
# ------------------------------------ data -----------------------------------#
# -----------------------------------------------------------------------------#

dataset = get_imagenet_dataset(transform, subset_size=25)
loader = DataLoader(dataset, batch_size=10)
def input_extractor(x): return x[0].to(device)
def target_extractor(x): return x[1].to(device)

# -----------------------------------------------------------------------------#
# ---------------------------------- explain ----------------------------------#
# -----------------------------------------------------------------------------#


def input_visualizer(x):
    return denormalize_image(x, transform.mean, transform.std)


def target_visualizer(x):
    return dataset.dataset.idx_to_label(x.item())


experiment = project.create_auto_experiment(
    model,
    loader,
    name='resnet_experiment',
    input_extractor=input_extractor,
    target_extractor=target_extractor,
    input_visualizer=input_visualizer,
    target_visualizer=target_visualizer,
)


model, transform = get_torchvision_model("vit_b_16")
model = model.to(device)

# -----------------------------------------------------------------------------#
# ------------------------------------ data -----------------------------------#
# -----------------------------------------------------------------------------#

dataset = get_imagenet_dataset(transform, subset_size=25)
loader = DataLoader(dataset, batch_size=10)
def input_extractor(x): return x[0].to(device)
def target_extractor(x): return x[1].to(device)

# -----------------------------------------------------------------------------#
# ---------------------------------- explain ----------------------------------#
# -----------------------------------------------------------------------------#


experiment = project.create_auto_experiment(
    model,
    loader,
    name='vit_experiment',
    input_extractor=input_extractor,
    target_extractor=target_extractor,
    input_visualizer=input_visualizer,
    target_visualizer=target_visualizer,
)

project.get_server().serve(debug=True, host='0.0.0.0', port=5001)
