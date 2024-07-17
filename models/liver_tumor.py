import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNet50(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet50(num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        x = x[:, 0, ...][:, None, ...]
        output = self.model(x)
        # output = F.log_softmax(output, dim=-1)
        # output = F.relu(output)
        # print('output:', output)
        return output
