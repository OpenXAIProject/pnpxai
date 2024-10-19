import torch
import math, numbers
import torch.nn as nn
import torch.nn.functional as F
import os, sys, inspect
import numpy as np
import cv2 as cv

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

try:
    import ResizeRight.resize_right as resize_right
    import ResizeRight.interp_methods as interp_methods
except Exception as err:
    print(str(err))

class BackgroundSubtractionWrapper(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, model, channels, kernel_size, sigma, device, dim=2, output_image=False):
        super(BackgroundSubtractionWrapper, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        self.device = device
        self.model = model
        self.output_image = output_image
        #self.device_ids = model.device_ids

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel.to(device))
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def _cut_image(self, img_ch, divisor):
        img = img_ch.permute(0, 2, 3, 1).clone()
        _, h, w, c = img.shape

        radius = int(h / 2)
        b = np.zeros((h, w, c))
        cv.circle(np.asarray(b), (radius, radius), int(radius * 0.9),
                                  (1, 1, 1), -1, 8, 0)
        print(b.shape)
        b = torch.tensor(b).to(self.device).unsqueeze(0).permute(0, 3, 1, 2).float()
        img_ch = img_ch * b + 128/divisor * (1-b)

        return img_ch

    def forward(self, input_src):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        if self.output_image:
            input_src = input_src.unsqueeze(0)


        device = input_src.device
        input_src = input_src * 255
        #print(input_src.shape)
        #print(input_src.dtype)
        input_ = F.pad(input_src, (3, 4, 3, 4), mode='reflect')
        ret = 4*input_src - 4*self.conv(input_, weight=self.weight.to(device), groups=self.groups) + 128
        #ret = (ret - ret.min())
        #ret = ret / ret.max()
        #ret /= ret.sum(0).expand_as(ret)
        #ret[torch.isnan(ret)]=0
        #return self.model(self._cut_image((ret/255).clip(0, 1), 255))
        if self.output_image:
            return (ret/255).clip(0, 1).squeeze(0)
        else:
            #return self.model(self._cut_image((ret / 255).clip(0, 1), 255))
            return self.model((ret/255).clip(0, 1))

class BiTMWrapper(torch.nn.Module):
    def __init__(self, model, mean, std, size):
        super().__init__()

        mean = None if mean is None else mean[..., None, None]
        std = None if std is None else std[..., None, None]

        #self.train(model.training)

        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.size = size
        #self.resize_layer = resize_right.ResizeLayer([1, 3, 32, 32], scale_factors=None,
        #                                             out_shape=[1, 3, self.size, self.size],
        #                                             interp_method=interp_methods.cubic, support_sz=None,
        #                                             antialiasing=True)

    def forward(self, x):
        if self.mean is not None and self.std is not None:
            x_normalized = (x - self.mean)/self.std
        else:
            x_normalized = x
        if self.size is not None:
            x_normalized = resize_right.resize(x_normalized, scale_factors=None,
                                               out_shape=[x_normalized.shape[0], 3, self.size, self.size],
                     interp_method=interp_methods.cubic, support_sz=None,
                     antialiasing=True)  ##F.interpolate(x_normalized, size=self.size)
        return self.model(x_normalized)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict()