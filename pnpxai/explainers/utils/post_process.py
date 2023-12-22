from typing import Literal
import torch
import numpy as np


def postprocess_attr(
    attr: torch.Tensor,
    sign: Literal["absolute", "positive", "negative"] = 'absolute',
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

    # Step 1: Sum over the channels to get a tensor of shape (batch, width, height)
    summed_tensor = attr.sum(dim=-1)

    # Step 2: Normalize each batch item independently
    # Find the max and min values for each item in the batch
    max_values = summed_tensor.view(summed_tensor.shape[0], -1).max(dim=1)[0].unsqueeze(1).unsqueeze(2)
    min_values = summed_tensor.view(summed_tensor.shape[0], -1).min(dim=1)[0].unsqueeze(1).unsqueeze(2)

    # Perform the normalization
    postprocessed = (summed_tensor - min_values) / (max_values - min_values)
    # postprocessed = torch.flip(postprocessed, [1])

    return postprocessed.cpu().detach().numpy()
