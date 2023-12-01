import torch
from torch.utils.data import DataLoader

from pnpxai.utils import set_seed
from pnpxai import Project
from pnpxai.visualizer.proc_manager.client import Client

from helpers import get_imagenet_dataset, get_torchvision_model


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

set_seed(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------#
# ----------------------------------- model -----------------------------------#
# -----------------------------------------------------------------------------#

model, transform = get_torchvision_model("resnet18")
model = model.to(device)

# -----------------------------------------------------------------------------#
# ------------------------------------ data -----------------------------------#
# -----------------------------------------------------------------------------#

dataset = get_imagenet_dataset(transform, subset_size=100)
loader = DataLoader(dataset, batch_size=8)
def input_extractor(x): return x[0].to(device)
def target_extractor(x): return x[1].to(device)

# -----------------------------------------------------------------------------#
# ---------------------------------- explain ----------------------------------#
# -----------------------------------------------------------------------------#


project = Project('test_project')
experiment = project.create_auto_experiment(
    model,
    loader,
    name='test_experiment',
    input_extractor=input_extractor,
    target_extractor=target_extractor
)
client = Client()
client.connect_to_or_start_server()
client.set_project(project.name, project)

print(len(experiment.runs))
