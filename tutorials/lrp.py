import os

import torch
import torchvision

from pnpxai.explainers import LRP

def get_model(model_name):
    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    model = torchvision.models.get_model(model_name, weights=weights).eval()
    transform = weights.transforms()
    return model, transform

def get_images(num_images, transform):
    IMG_DIR = "./tutorials/data/ImageNet/samples/"
    imgs = torch.stack([
        transform(torchvision.io.read_image(os.path.join(IMG_DIR, fnm)))
        for fnm in os.listdir(IMG_DIR)[:num_images]
    ])
    return imgs

model, transform = get_model("vit_b_16")
inputs = get_images(num_images=4, transform=transform)
targets = model(inputs).argmax(1)

explainer = LRP(model)
attributions = explainer.attribute(inputs, targets)
print(attributions.shape)