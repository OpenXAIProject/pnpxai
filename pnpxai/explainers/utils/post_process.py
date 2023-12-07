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
        pass
        # raise NotImplementedError

    postprocessed = attr.transpose(-1, -3)\
        .transpose(-2, -3).sum(dim=-1)

    aggr_shape = (-3, -2, -1)

    attr_max = torch.amax(postprocessed, dim=aggr_shape)
    attr_min = torch.amin(postprocessed, dim=aggr_shape)
    postprocessed = (postprocessed - attr_min) / (attr_max - attr_min)
    return postprocessed.cpu().detach().numpy()
