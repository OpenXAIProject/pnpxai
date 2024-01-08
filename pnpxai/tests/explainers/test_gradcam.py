import pytest
import torch
from torch import nn

from pnpxai.explainers import GradCam

INPUT_SIZE = (3, 2, 2)
NUM_CLASSES = 2

class ToyCNN(nn.Module):
    def __init__(self, with_pool=True):
        super().__init__()
        self.with_pool = with_pool
        self.conv = nn.Conv2d(INPUT_SIZE[0], 1, INPUT_SIZE[1]-1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(1, 1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(4, NUM_CLASSES)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.with_pool:
            x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

    @property
    def target_layer(self):
        if self.with_pool:
            return self.relu
        return None


class TestGradCam:
    def test_find_target_layer(self):
        model = ToyCNN()
        explainer = GradCam(model)
        assert explainer.source.layer is model.target_layer
    
    def test_find_target_layer_raises_without_pool(self):
        model_without_pool = ToyCNN(with_pool=False)
        with pytest.raises(AssertionError):
            explainer = GradCam(model_without_pool)
    
    def test_attribute(self):
        model = ToyCNN()
        input = torch.randn(1, *INPUT_SIZE)
        explainer = GradCam(model)
        attr = explainer.attribute(input, 0)
