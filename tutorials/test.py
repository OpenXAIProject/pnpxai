from torch.utils.data import DataLoader

from pnpxai.utils import set_seed
from pnpxai.explainers import LRPUniformEpsilon

from helpers import get_imagenet_dataset, get_torchvision_model

set_seed(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model, dataset, and explainer
model, transform = get_torchvision_model("resnet18")
model = model.to(device)
explainer = LRPUniformEpsilon(model=model, epsilon=1e-6, n_classes=1000)

dataset = get_imagenet_dataset(transform=transform, subset_size=8)
loader = DataLoader(dataset, batch_size=8)
inputs, targets = next(iter(loader))
inputs, targets = inputs.to(device), targets.to(device)

# make explanation
attrs = explainer.attribute(inputs, targets)

print(attrs)