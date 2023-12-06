import torch
from skimage.segmentation import felzenszwalb

def get_default_feature_mask(inputs: torch.Tensor, device):
    feature_mask = [
        torch.tensor(felzenszwalb(input.permute(1,2,0).detach().cpu().numpy(), scale=250))
        for input in inputs
    ]
    return torch.LongTensor(torch.stack(feature_mask)).to(device)