from typing import Literal
import torch
import numpy as np

def postprocess_attr(
        attr: torch.Tensor,
        sign: Literal["absolute", "positive", "negative"] = 'absolute'
    ) -> np.array:
    if sign == 'absolute':
        attr = torch.abs(attr)
    elif sign == 'positive':
        attr = torch.nn.functional.relu(attr)
    elif sign == 'negative':
        attr = -torch.nn.functional.relu(-attr)
    else:
        raise NotImplementedError
    postprocessed = attr.permute((1, 2, 0)).sum(dim=-1)
    attr_max = torch.max(postprocessed)
    attr_min = torch.min(postprocessed)
    postprocessed = (postprocessed - attr_min) / (attr_max - attr_min)
    return postprocessed.cpu().detach().numpy()