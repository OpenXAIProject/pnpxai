import torch
from torch.utils.data import DataLoader, Subset

from pnpxai.utils import set_seed
from pnpxai import Project
from pnpxai.explainers import RAP

from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image


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
dataset = Subset(dataset, list(range(25)))
loader = DataLoader(dataset, batch_size=10)
def input_extractor(x): return x[0].to(device)
def target_extractor(x): return x[1].to(device)

# -----------------------------------------------------------------------------#
# ---------------------------------- explain ----------------------------------#
# -----------------------------------------------------------------------------#


def input_visualizer(x): return denormalize_image(x, transform.mean, transform.std)


project = Project('test_project')
experiment = project.create_experiment(
    model,
    loader,
    name='test_experiment',
    explainers=[RAP(model)],
    input_extractor=input_extractor,
    target_extractor=target_extractor,
    input_visualizer=input_visualizer
)
experiment.run()
run = experiment.runs[0]
vis = run.get_flattened_visualizations(experiment.task)
vis[0].show()