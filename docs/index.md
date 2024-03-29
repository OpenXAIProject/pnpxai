# pnpxai: Plug-and-Play Explainable AI

pnpxai is a Python package that provides a modular and easy-to-use framework for explainable artificial intelligence (XAI). It allows users to apply various XAI methods to their own models and datasets, and visualize the results in an interactive and intuitive way.

## Features

- [**Detector**](detector): The detector module provides automatic detection of AI models implemented in PyTorch.
- [**Evaluator**](evaluator): The evaluator module provides various ways to evaluate and compare the performance and explainability of AI models, such as [correctness](api/evaluator), and [continuity](api/evaluator).
- **Explainers**: The explainers module contains a collection of state-of-the-art XAI methods that can generate global or local explanations for any AI model, such as [SHAP](https://github.com/openxaiproject/pnpxai/explainers/kernel_shap), [LIME](https://github.com/openxaiproject/pnpxai/explainers/lime), [IG](https://github.com/openxaiproject/pnpxai/explainers/integrated_gradients), [LRP](https://github.com/openxaiproject/pnpxai/explainers/lrp), and [RAP](https://github.com/openxaiproject/pnpxai/explainers/rap).
- [**Recommender**](recommender): The recommender module offers a recommender system that can suggest the most suitable XAI methods for a given model and dataset, based on the user’s preferences and goals.
- **Visualizer**: The visualizer module enables users to interact with and explore the results of XAI methods in a web-based dashboard, using various charts, graphs, and widgets.

## Project Core API

* [**Project**](api/core/project.md): module, designed for combining semantically similar experiments for side-to-side comparison
* [**Experiment**](api/core/experiment.md): module, responsible for data manipulation, model explanation and explanations' evaluation
* [**Auto Experiment**](api/core/auto_experiment.md): module, responsible for data manipulation, model explanation and explanations' evaluation

## Installation

To install `pnpxai`, run the following command:

```bash
# Command lines for installation
pip install -e .
```

## Getting Started

To use `pnpxai`, you need to import the package and its modules in your Python script. Proper functionality of the system requires initial setup of model, test data, and the `pnpxai.Project`.

```python
import torch
from torcu.utils.data import DataLoader

from pnpxai import Project

# Bring your model
model = ...

# Prepare your data
dataset = ...
loader = DataLoader(dataset, batch_size=...)
```

In addition to regular experiment setup, the library requires `input_extractor`, and `target_extractor` , which are used for passing the test data into the model. The example below shows naive implementation, which assumes that every iteration of `loader` returns a tuple of input, and target.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def input_extractor(x): return x[0].to(device)
def target_extractor(x): return x[1].to(device)
```

Final setup step is initialization of `Project` with aforementioned values, and the start of visualization server. Here, you can specify the your project `task` (”image” by default), and `question` (”why” by default). Additionally, `pnpxai.Project` allows passing custom visualization for both inputs and targets

Complete backbone of initialization code can be represented as follows:

```python
import torch
from torcu.utils.data import DataLoader

from pnpxai import Project

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
project = Project('YOUR_XAI_PROJECT_NAME')
experiment = project.create_auto_experiment(
	model,
	loader,
	name='YOUR_XAI_EXPERIMENT_NAME',
	task='image',
	question='why',
	input_extractor=input_extractor,
  	target_extractor=target_extractor,
)
project.get_server().serve()
```

`project.create_auto_experiment(...)` method is guided by `pnpxai.recommender.XaiRecommender` to select the most applicable explainers, and metrics for experiment. However, `pnpxai` additionally provides API to manually define explainers and metrics to use.

```python
import torch
from torcu.utils.data import DataLoader

from pnpxai import Project
from pnpxai.explainers import LRP
from pnpxai.evaluator import LRP

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
project = Project('YOUR_XAI_PROJECT_NAME')
experiment = project.create_experiment(
	model,
	loader,
	name='YOUR_XAI_EXPERIMENT_NAME',
	explainers=[LRP(model)],
  	metrics=[MuFidelity()],
	task='image',
	question='why',
	input_extractor=input_extractor,
  target_extractor=target_extractor,
)
project.get_server().serve()
```