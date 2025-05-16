# PnPXAI: Plug-and-Play Explainable AI
<hr>
<div align='center'>
    <img src="assets/pnpxai_logo_horizontal.png">
</div>

`PnPXAI` is a Python package that provides a modular and easy-to-use framework for explainable artificial intelligence (XAI). It allows users to apply various XAI methods to their own models and datasets, and visualize the results in an interactive and intuitive way.

<hr>

## Features

- [**Detector**](detector.md): The detector module provides automatic detection of AI models implemented in PyTorch.
- [**Evaluator**](evaluator.md): The evaluator module provides various ways to evaluate and compare the performance and explainability of AI models with the categorized evaluation properties of correctness ([fidelity](api/evaluator/metrics.md/#pnpxai.evaluator.metrics.mu_fidelity.MuFidelity), [area between perturbation curves](api/evaluator/metrics/#pnpxai.evaluator.metrics.pixel_flipping.AbPC)), continuity ([sensitivity](api/evaluator/metrics.md#pnpxai.evaluator.metrics.sensitivity.Sensitivity)), and compactness ([complexity](api/evaluator/metrics.md#pnpxai.evaluator.metrics.complexity.Complexity)).
- [**Explainers**](explainer.md): The explainers module contains a collection of state-of-the-art XAI methods that can generate global or local explanations for any AI model, such as:
	- Perturbation-based ([SHAP](api/explainer/kernel_shap), [LIME](api/explainer/lime))
	- Relevance-based ([IG](api/explainer/ig), [LRP](api/pnpxai/explainer/lrp), and [RAP](api/explainer/rap))
	- CAM-based ([GradCAM](api/explainer/grad_cam), [Guided GradCAM](api/explainer/guided_grad_cam))
	- Gradient-based ([SmoothGrad](api/explainer/smooth_grad), [VarGrad](api/explainer/var_grad), [Gradient &times; Input](api/explainer/grad_x_input))
- [**Recommender**](recommender.md): The recommender module offers a recommender system that can suggest the most suitable XAI methods for a given model and dataset, based on the user’s preferences and goals.
- [**Optimizer**](optimizer.md): The optimizer module is finds the best hyperparameter options, given a user-specified metric.

## Project Core API

* [**Experiment**](api/core/experiment.md): module, responsible for data manipulation, model explanation and explanations' evaluation
* [**Auto Explanation**](api/core/auto_explanation.md): module, responsible for data manipulation, model explanation and explanations' evaluation

<hr>

## Installation

To install `pnpxai` from `pip`, run the following command:

```bash
pip install pnpxai
```

To install `pnpxai` from `GitHub`, run the following commands:

```bash
git clone git@github.com:OpenXAIProject/pnpxai.git
cd pnpxai
pip install -e .
```

<hr>

## Getting Started

### Auto Explanation

To use `pnpxai`, you need to import the package and its modules in your Python script. Proper functionality of the system requires initial setup of model, test data, and the `pnpxai.AutoExplanationForImageClassification`.

The explanation module can be specified according to a modality, which fits best for the user's task. Specifically, `pnpxai` offers [`AutoExplanationForImageClassification`](api/core/auto_explanation.md/#pnpxai.core.experiment.auto_explanation.AutoExplanationForImageClassification), [`AutoExplanationForTextClassification`](api/core/auto_explanation.md/#pnpxai.core.experiment.auto_explanation.AutoExplanationForTextClassification), [`AutoExplanationForTSClassification`](api/core/auto_explanation.md/#pnpxai.core.experiment.auto_explanation.AutoExplanationForTSClassification),
[`AutoExplanationForVisualQuestionAnswering`](api/core/auto_explanation.md/#pnpxai.core.experiment.auto_explanation.AutoExplanationForVisualQuestionAnswering), which can be utilized for image, test, time series, and a combination of image and text modalities respectively.

```python
import torch
from torch.utils.data import DataLoader

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
from torch.utils.data import DataLoader

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
from torch.utils.data import DataLoader

from pnpxai import Experiment
from pnpxai.core.modality import ImageModality
from pnpxai.explainers import LRPEpsilonPlus
from pnpxai.evaluator.metrics import MuFidelity
from pnpxai.explainers.utils.postprocess import Identity

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
postprocessor = Identity()
modality = ImageModality()

experiment = Experiment(
    model=model,
    data=loader,
    modality=ImageModality(),
    explainers=[explainer],
    postprocessors=[Identity()],
    metrics=[metric],
    input_extractor=lambda x: x[0].to(device),
    label_extractor=lambda x: x[-1].to(device),
    target_extractor=lambda outputs: outputs.argmax(-1).to(device)
)
```

<hr>

## Tutorials
- [Image Classification](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/auto_explanation_imagenet_example.py)
- [Text Classification](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/auto_explanation_imdb_example.py)
- [Time Series Classification](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/auto_explanation_ts_example.py)
- [Visual Question Answering](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/auto_explanation_vqa_example.py)
- [Evaluator](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/evaluator.py)
- [ImageNet Example All Explainers](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/imagenet_example_all_explainers.md)
- [ImageNet Example All Metrics](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/imagenet_example_all_metrics.md)
- [Free MCG](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/gfgp_tutorial.py) [[Reference](https://arxiv.org/abs/2411.15265)] 

<hr>

## Use Cases

Medical Domain Explainability

- Counterfactual Explanation ([LEAR (Learn-Explain-Reinforce)](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/LEAR_example_all_explainers.md)) for Alzheimer’s Disease Diagnosis, a joint work with Research Task 2 (PI Bohyung Han, Seoul National University) [[Reference](https://ieeexplore.ieee.org/document/9854196)]

- Attribution-based Explanation for [Dysarthria Diagnosis](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/xai_ddk_pnpxai_example.md), a joint work with Research Task 3 (PI Myoung-Wan Koo, Sogang University)


LLM Trsutworthiness

- Evaluating the Factuality of Korean Text Generated by LLMs ([KorFactScore (Korean Factual precision in atomicity Score)](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/fact_score_example_korfactscore.py)), a joint work with Research Task 4 (PI Kyongman Bae, ETRI)
 [[Reference](https://github.com/ETRI-XAINLP/KorFactScore)]

<hr>

## Acknowledgements

> This research was initiated by KAIST XAI Center and conducted in collaboration with multiple institutions, including Seoul National University, Korea University, Sogang University, and ETRI.
We are grateful for the grant from the Institute of Information & communications Technology Planning & Evaluation (IITP) (No.RS-2022-II220984).

<hr>

## Citation

If you find this repository useful in your research, please consider citing our paper:

```
@article{kim2025pnpxai,
  title={PnPXAI: A Universal XAI Framework Providing Automatic Explanations Across Diverse Modalities and Models},
  author=author={Kim, Seongun and Kim, Sol A and Kim, Geonhyeong and Menadjiev, Enver and Lee, Chanwoo and Chung, Seongwook and Kim, Nari and Choi, Jaesik},
  journal={arXiv preprint arXiv:2505.10515},
  year={2025}
}
```

<hr>

## License

PnP XAI is released under Apache license 2.0. See [LICENSE](https://github.com/OpenXAIProject/pnpxai/tree/main/LICENSE) for additional details.
