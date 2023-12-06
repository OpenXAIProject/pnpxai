import torch
import torchvision

from open_xai import Project
from open_xai.explainers import LRPEpsilon, LRPGamma

model = torchvision.models.get_model("inception_v3").eval()
inputs = torch.randn(1, 3, 224, 224)
target = model(inputs).argmax(1).item()

proj = Project("test_lrp")

proj.explain(LRPEpsilon(model, epsilon=1e-6))
proj.explain(LRPGamma(model, gamma=.25))

for experiment in proj.experiments:
    experiment.run(inputs, target)

print("LRPEpsilon")
print(proj.experiments[0].runs[0].outputs, "\n")
print("LRPGamma")
print(proj.experiments[1].runs[0].outputs)
