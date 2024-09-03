# pnpxai: Plug-and-Play Explainable AI

pnpxai is a Python package that provides a modular and easy-to-use framework for explainable artificial intelligence (XAI). It allows users to apply various XAI methods to their own models and datasets, and visualize the results in an interactive and intuitive way.

## Features

- [**Detector**](detector.md): The detector module provides automatic detection of AI models implemented in PyTorch.
- [**Evaluator**](evaluator.md): The evaluator module provides various ways to evaluate and compare the performance and explainability of AI models, such as [complexity](api/evaluator/metrics.md#pnpxai.evaluator.metrics.complexity.Complexity), [fidelity](api/evaluator/metrics.md/#pnpxai.evaluator.metrics.mu_fidelity.MuFidelity), [sensitivity](api/evaluator/metrics.md#pnpxai.evaluator.metrics.sensitivity.Sensitivity), and [area between perturbation curves](api/evaluator/metrics/#pnpxai.evaluator.metrics.pixel_flipping.AbPC).
- **Explainers**: The explainers module contains a collection of state-of-the-art XAI methods that can generate global or local explanations for any AI model, such as:
	- Perturbation-based ([SHAP](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/kernel_shap.py), [LIME](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/lime.py))
	- Relevance-based ([IG](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/integrated_gradients.py), [LRP](https://github.com/openxaiproject/pnpxai/explainers/lrp), and [RAP](https://github.com/openxaiproject/pnpxai/explainers/rap), [GuidedBackprop](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/guided_backprop.py))
	- CAM-based ([GradCAM](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/grad_cam.py), [Guided GradCAM](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/guided_grad_cam.py))
	- Gradient-based ([SmoothGrad](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/smooth_grad.py), [VarGrad](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/var_grad.py), [FullGrad](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/full_grad.py), [Gradient &times; Input](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/grad_x_input.py))
- [**Recommender**](recommender.md): The recommender module offers a recommender system that can suggest the most suitable XAI methods for a given model and dataset, based on the user’s preferences and goals.
- [**Optimizer**](optimizer.md): The optimizer module is finds the best hyperparameter options, given a user-specified metric.

## Project Core API

* [**Experiment**](api/core/experiment.md): module, responsible for data manipulation, model explanation and explanations' evaluation
* [**Auto Explanation**](api/core/auto_explanation.md): module, responsible for data manipulation, model explanation and explanations' evaluation

## Installation

To install `pnpxai`, run the following command:

```bash
# Command lines for installation
pip install -e .
```

## Getting Started

### Auto Explanation

To use `pnpxai`, you need to import the package and its modules in your Python script. Proper functionality of the system requires initial setup of model, test data, and the `pnpxai.AutoExplanationForImageClassification`.

The explanation module can be specified according to a modality, which fits best for the user's task. Specifically, `pnpxai` offers [`AutoExplanationForImageClassification`](api/core/auto_explanation.md/#pnpxai.core.experiment.auto_explanation.AutoExplanationForImageClassification), [`AutoExplanationForTextClassification`](api/core/auto_explanation.md/#pnpxai.core.experiment.auto_explanation.AutoExplanationForTextClassification), [`AutoExplanationForTSClassification`](api/core/auto_explanation.md/#pnpxai.core.experiment.auto_explanation.AutoExplanationForTSClassification),
[`AutoExplanationForVisualQuestionAnswering`](api/core/auto_explanation.md/#pnpxai.core.experiment.auto_explanation.AutoExplanationForVisualQuestionAnswering), which can be utilized for image, test, time series, and a combination of image and text modalities respectively.

```python
import torch
from torcu.utils.data import DataLoader

from pnpxai import AutoExplanationForImageClassification

# Bring your model
model = ...

# Prepare your data
dataset = ...
loader = DataLoader(dataset, batch_size=...)
```

In addition to regular experiment setup, the library requires `input_extractor`, `target_extractor`, and `label_extractor`, which are used for passing the test data into the model. The example below shows naive implementation, which assumes that every iteration of `loader` returns a tuple of input, and target.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def input_extractor(x): return x[0].to(device)
def label_extractor(x): return x[1].to(device)
def target_extractor(outputs): outputs.argmax(-1).to(device)
```

Final setup step is initialization of `AutoExplanationForImageClassification` with aforementioned values, and the start of optimization process. To start optimization process, it is required to choose desired data ids, as well as, explainer, and metric from the list of suggested ones.

```python
experiment = AutoExplanationForImageClassification(
	model,
	loader,
	input_extractor=input_extractor,
  	label_extractor=label_extractor,
  	target_extractor=target_extractor,
	target_labels=False,
)
optimized = experiment.optimize(
    data_ids=range(16),
    explainer_id=2,
    metric_id=1,
    direction='maximize', # less is better
    sampler='tpe', # Literal['tpe','random']
    n_trials=50, # by default, 50 for sampler in ['random', 'tpe'], None for ['grid']
    seed=42, # seed for sampler: by default, None
)
```

Complete backbone of initialization code can be represented as follows:

```python
import torch
from torcu.utils.data import DataLoader

from pnpxai import AutoExplanationForImageClassification

# Bring your model
model = ...

# Prepare your data
dataset = ...
loader = DataLoader(dataset, batch_size=...)
def input_extractor(x):
	...
def target_extractor(x):
	...

# Auto-explanation
experiment = AutoExplanationForImageClassification(
	model,
	loader,
	input_extractor=input_extractor,
  	label_extractor=label_extractor,
  	target_extractor=target_extractor,
	target_labels=False,
)
optimized = experiment.optimize(
    data_ids=range(16),
    explainer_id=2,
    metric_id=1,
    direction='maximize', # less is better
    sampler='tpe', # Literal['tpe','random']
    n_trials=50, # by default, 50 for sampler in ['random', 'tpe'], None for ['grid']
    seed=42, # seed for sampler: by default, None
)
```
### Manual Setup

`AutoExplanationForImageClassification` class is guided by [`pnpxai.XaiRecommender`](recommender.md) to select the most applicable explainers, and metrics for experiment. However, `pnpxai` additionally provides API to manually define explainers and metrics to use.

Here, users are asked to manually define modalities, in order to enable modality-dependent control flow. The `pnpxai` package comes with a set of predefined modalities, namely [`ImageModality`](api/core/modality.md/#pnpxai.core.modality.modality.ImageModality), [`TextModality`](api/core/modality.md/#pnpxai.core.modality.modality.TextModality), [`TimeSeriesModality`](api/core/modality.md/#pnpxai.core.modality.modality.TimeSeriesModality). However, API also enables extension possiblity with the help of a [`Modality`](api/core/modality.md/#pnpxai.core.modality.modality.Modality) base class.

```python
import torch
from torcu.utils.data import DataLoader

from pnpxai import Experiment
from pnpxai.core.modality import ImageModality
from pnpxai.explainers import LRPEpsilonPlus
from pnpxai.evaluator.metrics import MuFidelity

# Bring your model
model = ...

# Prepare your data
dataset = ...
loader = DataLoader(dataset, batch_size=...)
def input_extractor(x):
	...
def label_extractor(x):
	...
def target_extractor(x):
	...

# Auto-explanation
explainer = LRPEpsilonPlus(model)
metric = MuFidelity(model, explainer)

experiment = Experiment(
	explainers=[explainer],
  	metrics=[metric],
	input_extractor=input_extractor,
	label_extractor=label_extractor,
  	target_extractor=target_extractor,
)
```