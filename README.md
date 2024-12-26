# PnPXAI: Plug-and-Play Explainable AI

<div align='center'>
    <img src="assets/pnpxai_logo_horizontal.png">
</div>

[PnPXAI](https://openxaiproject.github.io/pnpxai/) is a Python package that provides a modular and easy-to-use framework for explainable artificial intelligence (XAI). It allows users to apply various XAI methods to their own models and datasets, and visualize the results in an interactive and intuitive way.

## Features

- [**Detector**](pnpxai/core/detector/): The detector module provides automatic detection of AI models implemented in PyTorch.
- [**Evaluator**](pnpxai/evaluator/metrics/): The evaluator module provides various ways to evaluate and compare the performance and explainability of AI models with the categorized evaluation properties of correctness ([fidelity](pnpxai/evaluator/metrics/mu_fidelity.py), [area between perturbation curves](pnpxai/evaluator/metrics/pixel_flipping.py)), continuity ([sensitivity](pnpxai/evaluator/metrics/sensitivity.py)), and compactness ([complexity](pnpxai/evaluator/metrics/complexity.py)).
- **Explainers**: The explainers module contains a collection of state-of-the-art XAI methods that can generate global or local explanations for any AI model, such as:
	- Perturbation-based ([SHAP](pnpxai/explainers/kernel_shap.py), [LIME](pnpxai/explainers/lime.py))
	- Relevance-based ([IG](pnpxai/explainers/integrated_gradients.py), [LRP](pnpxai/explainers/lrp.py), [RAP](pnpxai/explainers/rap), [GuidedBackprop](pnpxai/explainers/guided_backprop.py))
	- CAM-based ([GradCAM](pnpxai/explainers/grad_cam.py), [Guided GradCAM](pnpxai/explainers/guided_grad_cam.py))
	- Gradient-based ([SmoothGrad](pnpxai/explainers/smooth_grad.py), [VarGrad](pnpxai/explainers/var_grad.py), [FullGrad](pnpxai/explainers/full_grad.py), [Gradient &times; Input](pnpxai/explainers/grad_x_input.py))
- [**Recommender**](pnpxai/core/recommender): The recommender module offers a recommender system that can suggest the most suitable XAI methods for a given model and dataset, based on the user’s preferences and goals.
- [**Optimizer**](pnpxai/evaluator/optimizer): The optimizer module is finds the best hyperparameter options, given a user-specified metric.

## Installation

To install `pnpxai`, run the following command:

```bash
# Command lines for installation
pip install -e .
```

## Getting Started

This guide explains how to automatically explain your own models and datasets using the provided Python script. The complete code can be found [here](tutorials/auto_explanation_imagenet_example.py).

1. **Setup**: The setup involves setting a random seed for reproducibility and defining the device for computation (CPU or GPU). 
    
    ```python
    import torch
    from pnpxai.utils import set_seed
    
    # Set the seed for reproducibility
    set_seed(seed=0)
    
    # Determine the device based on the availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```
    
2. **Create Experiments**: An experiment is an instance for explaining a specific model and dataset. Before creating an experiment, define the model and dataset to be explained.

    **Automatic explainer selection**: The `AutoExplanationForImageClassification` method automatically selects the most applicable explainers and metrics based on the model architecture using `pnpxai.XaiRecommender`.
        
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
        
    **Manual explainer selection**: Alternatively, you can manually specify the desired explanation method and evaluation metric using `Experiment`.
        
    ```python
    from pnpxai.core.modality import ImageModality
    from pnpxai.explainers import LRPEpsilonPlus
    from pnpxai.evaluator.metrics import MuFidelity
    
    explainer = LRPEpsilonPlus(model)
    metric = MuFidelity(model, explainer)
    modality = ImageModality()

    experiment = Experiment(
        model,
        loader,
        modality,
        explainers=[explainer],
        metrics=[metric],
        input_extractor=input_extractor,
        label_extractor=label_extractor,
        target_extractor=target_extractor,
    )
    ```


## Tutorials
- [Image Classification](tutorials/auto_explanation_imagenet_example.py)
- [Text Classification](tutorials/auto_explanation_imdb_example.py)
- [Time Series Classification](tutorials/auto_explanation_ts_example.py)
- [Visual Question Answering](tutorials/auto_explanation_vqa_example.py)
- [Evaluator](tutorials/evaluator.py)
- [ImageNet Example All Explainers](tutorials/imagenet_example_all_explainers.md)
- [ImageNet Example All Metrics](tutorials/imagenet_example_all_metrics.md)

## [Demo](https://openxaiproject.github.io/pnpxai/demo/)

## Documentation

The [Documentation](https://openxaiproject.github.io/pnpxai/) contains the API reference for all of the functionality of the framework. Primarily, high-level modules of the framework include: 
- Detector
- Explainer
- Recommender
- Evaluator
- Optimizer

## Acknowledgements

> This research was initiated by KAIST XAI Center and conducted in collaboration with multiple institutions, including Seoul National University, Korea University, Sogang University, and ETRI.
We are grateful for the grant from the Institute of Information & communications Technology Planning & Evaluation (IITP) (No.RS-2022-II220984).

## License

PnP XAI is released under Apache license 2.0. See [LICENSE](LICENSE) for additional details.
