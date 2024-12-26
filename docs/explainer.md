# Explainer <small>[[source](api/explainer/base.md)]</small>

The Explainer module serves as a basis for all explainers implemented within the framework. When it is necessary to implement a custom explainer, the base class needs to be extended to enable the support of the framework's functionality. The method accumulates crucial attributes and methods, making it visible to the [Recommender](recommender.md), and [Optimizer](optimizer.md) modules.

## List of available explainers:
| Explainer                                                                                                          | Supported Modules                                           | Target Modalities                              |
| ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- | ---------------------------------------------- |
| [GradCam](api/explainer/grad_cam.md)                                                                               | Convolution                                                 | Vision, Time Series                            |
| [GuidedGradCam](api/explainer/guided_grad_cam.md)                                                                  | Convolution                                                 | Vision, Time Series                            |
| [Gradient](api/explainer/gradient.md)                                                                              | Linear, Convolution, Recurrent, Transformer                 | Vision, Language, Time Series                  |
| [GradientXInput](api/explainer/grad_x_input.md)                                                                    | Linear, Convolution, Recurrent, Transformer                 | Vision, Language, Time Series                  |
| [SmoothGrad](api/explainer/smooth_grad.md)                                                                         | Linear, Convolution, Recurrent, Transformer                 | Vision, Language, Time Series                  |
| [VarGrad](api/explainer/var_grad.md)                                                                               | Linear, Convolution, Recurrent, Transformer                 | Vision, Language, Time Series                  |
| [IntegratedGradients](api/explainer/ig.md)                                                                         | Linear, Convolution, Recurrent, Transformer                 | Vision, Language, Time Series                  |
| [LRPUniformEpsilon](api/explainer/lrp.md)                                                                          | Linear, Convolution, Recurrent, Transformer                 | Vision, Language, Time Series                  |
| [LRPEpsilonPlus](api/explainer/lrp.md)                                                                             | Linear, Convolution, Recurrent, Transformer                 | Vision, Language, Time Series                  |
| [LRPEpsilonGammaBox](api/explainer/lrp.md)                                                                         | Linear, Convolution, Recurrent, Transformer                 | Vision, Language, Time Series                  |
| [LRPEpsilonAlpha2Beta1](api/explainer/lrp.md)                                                                      | Linear, Convolution, Recurrent, Transformer                 | Vision, Language, Time Series                  |
| [RAP](api/explainer/rap.md)                                                                                        | Linear, Convolution, Recurrent, Transformer                 | Vision, Language, Time Series                  |
| [KernelShap](api/explainer/kernel_shap.md)                                                                         | Linear, Convolution, Recurrent, Transformer, Decision Trees | Vision, Language, Structured Data, Time Series |
| [Lime](api/explainer/lime.md)                                                                                      | Linear, Convolution, Recurrent, Transformer, Decision Trees | Vision, Language, Structured Data, Time Series |
| [AttentionRollout](api/explainer/attn_rollout.md#pnpxai.explainers.attention_rollout.AttentionRollout)             | Transformer                                                 | Vision, Language                               |
| [TransformerAttribution](api/explainer/attn_rollout.md#pnpxai.explainers.attention_rollout.TransformerAttribution) | Transformer                                                 | Vision, Language                               |

## Usage

```python
import torch
from torch.utils.data import DataLoader

from pnpxai.utils import set_seed
from pnpxai.explainers import LRPUniformEpsilon

from helpers import get_imagenet_dataset, get_torchvision_model

set_seed(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model, dataset, and explainer
model, transform = get_torchvision_model("resnet18")
model = model.to(device)
explainer = LRPUniformEpsilon(model=model, epsilon=1e-6, n_classes=1000)

dataset = get_imagenet_dataset(transform=transform, subset_size=8)
loader = DataLoader(dataset, batch_size=8)
inputs, targets = next(iter(loader))
inputs, targets = inputs.to(device), targets.to(device)

# make explanation
attrs = explainer.attribute(inputs, targets)

print(attrs)
```