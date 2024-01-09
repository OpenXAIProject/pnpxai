import torch
from torch import nn

TEST_IMAGE_TENSOR_SIZE = (3, 2, 2)
TEST_CLASSIFIER_OUTPUT_DIM = 2

def get_test_input_image(batch=True):
    img = torch.randn(*TEST_IMAGE_TENSOR_SIZE)
    if batch:
        return img.unsqueeze(0)
    return img

class _TestModelCNN(nn.Module):
    def __init__(self, with_pool=True):
        super().__init__()
        self.with_pool = with_pool

        _input_image = get_test_input_image(batch=False)
        _input_size = _input_image.shape
        self.conv = nn.Conv2d(_input_size[0], 1, _input_size[1]-1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(1, 1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(4, 2)
    
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
