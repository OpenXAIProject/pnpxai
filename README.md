# PnPXAI: Plug-and-Play Explainable AI

<div align='center'>
    <img src="https://raw.githubusercontent.com/OpenXAIProject/pnpxai/main/assets/pnpxai_logo_horizontal.png">
</div>

[PnPXAI](https://openxaiproject.github.io/pnpxai/) is a Python package that provides a modular and easy-to-use framework for explainable artificial intelligence (XAI). It allows users to apply various XAI methods to their own models and datasets, and visualize the results in an interactive and intuitive way.

## Features

- [**Detector**](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/core/detector): The detector module provides automatic detection of AI models implemented in PyTorch.
- [**Evaluator**](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/evaluator/metrics/): The evaluator module provides various ways to evaluate and compare the performance and explainability of AI models with the categorized evaluation properties of correctness ([fidelity](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/evaluator/metrics/mu_fidelity.py), [area between perturbation curves](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/evaluator/metrics/pixel_flipping.py)), continuity ([sensitivity](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/evaluator/metrics/sensitivity.py)), and compactness ([complexity](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/evaluator/metrics/complexity.py)).
- **Explainers**: The explainers module contains a collection of state-of-the-art XAI methods that can generate global or local explanations for any AI model, such as:
	- Perturbation-based ([SHAP](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/kernel_shap.py), [LIME](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/lime.py))
	- Relevance-based ([IG](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/integrated_gradients.py), [LRP](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/lrp.py), [RAP](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/rap), [GuidedBackprop](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/guided_backprop.py))
	- CAM-based ([GradCAM](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/grad_cam.py), [Guided GradCAM](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/guided_grad_cam.py))
	- Gradient-based ([SmoothGrad](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/smooth_grad.py), [VarGrad](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/var_grad.py), [FullGrad](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/full_grad.py), [Gradient &times; Input](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/explainers/grad_x_input.py))
- [**Recommender**](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/core/recommender): The recommender module offers a recommender system that can suggest the most suitable XAI methods for a given model and dataset, based on the user’s preferences and goals.
- [**Optimizer**](https://github.com/OpenXAIProject/pnpxai/tree/main/pnpxai/evaluator/optimizer): The optimizer module is finds the best hyperparameter options, given a user-specified metric.

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

## Getting Started

This guide explains how to automatically explain your own models and datasets using the provided Python script. The complete code can be found [here](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/auto_explanation_imagenet_example.py).

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
- [Image Classification](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/auto_explanation_imagenet_example.py)
- [Text Classification](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/auto_explanation_imdb_example.py)
- [Time Series Classification](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/auto_explanation_ts_example.py)
- [Visual Question Answering](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/auto_explanation_vqa_example.py)
- [Evaluator](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/evaluator.py)
- [ImageNet Example All Explainers](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/imagenet_example_all_explainers.md)
- [ImageNet Example All Metrics](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/imagenet_example_all_metrics.md)
- [Free MCG](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/gfgp_tutorial.py) [[Reference](https://arxiv.org/abs/2411.15265)] 

## Use Cases

Medical Domain Explainability

- Counterfactual Explanation ([LEAR (Learn-Explain-Reinforce)](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/LEAR_example_all_explainers.md)) for Alzheimer’s Disease Diagnosis, a joint work with Research Task 2 (PI Bohyung Han, Seoul National University) [[Reference](https://ieeexplore.ieee.org/document/9854196)]

- Attribution-based Explanation for [Dysarthria Diagnosis](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/xai_ddk_pnpxai_example.md), a joint work with Research Task 3 (PI Myoung-Wan Koo, Sogang University)


LLM Trsutworthiness

- Evaluating the Factuality of Korean Text Generated by LLMs ([KorFactScore (Korean Factual precision in atomicity Score)](https://github.com/OpenXAIProject/pnpxai/tree/main/tutorials/fact_score_example_korfactscore.py)), a joint work with Research Task 4 (PI Kyongman Bae, ETRI)
 [[Reference](https://github.com/ETRI-XAINLP/KorFactScore)]


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

## License

PnP XAI is released under Apache license 2.0. See [LICENSE](https://github.com/OpenXAIProject/pnpxai/tree/main/LICENSE) for additional details.
