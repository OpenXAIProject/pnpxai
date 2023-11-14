import torch

from pnpxai.detector import get_model_architecture

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).eval()
input = torch.randn(1, 3, 224, 224)
resnet_architecture = get_model_architecture(resnet, input)
print("[resnet18 from torchvision]")
print("resnet_architecture.is_convolution()", resnet_architecture.is_convolution())
print("resnet_architecture.is_transformer()", resnet_architecture.is_transformer(), "\n")