import torch
import torchvision

from open_xai import Project
from open_xai.explainers import GradCam

model = torchvision.models.get_model("inception_v3").eval()
inputs = torch.randn(1, 3, 224, 224)
target = model(inputs).argmax(1).item()

proj = Project('test_project')
exp = proj.explain(GradCam(model))
run = exp.run(inputs, target=target)

print(run.outputs)