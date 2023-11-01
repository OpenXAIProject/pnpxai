import torch
import torchvision
from skimage.segmentation import slic

from open_xai import Project
from open_xai.explainers import Lime

model = torchvision.models.get_model("inception_v3").eval()
inputs = torch.randn(1, 3, 224, 224)
target = model(inputs).argmax(1).item()
feature_mask = torch.Tensor(slic(
    inputs[0].permute(1, 2, 0).detach().numpy(),
    n_segments = 100,
    compactness = 10,
)).long()

proj = Project('lime_test')
exp = proj.explain(Lime(model))
run = exp.run(inputs, target=target, feature_mask=feature_mask)

print(run.outputs)