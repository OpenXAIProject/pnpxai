# PnPXAI: Plug-and-Play Explainable AI

<div align='center'>
    <img src="assets/pnpxai_logo_horizontal.png">
</div>

[PnPXAI](https://openxaiproject.github.io/pnpxai/) is a Python package that provides a modular and easy-to-use framework for explainable artificial intelligence (XAI). It allows users to apply various XAI methods to their own models and datasets, and visualize the results in an interactive and intuitive way.

## Features

- [**Detector**](pnpxai/detector): The detector module provides automatic detection of AI models implemented in PyTorch.
- [**Evaluator**](pnpxai/evaluator): The evaluator module provides various ways to evaluate and compare the performance and explainability of AI models, such as [correctness](pnpxai/evaluator/infidelity.py), and [continuity](pnpxai/evaluator/sensitivity.py).
- [**Explainers**](pnpxai/explainers): The explainers module contains a collection of state-of-the-art XAI methods that can generate global or local explanations for any AI model, such as [SHAP](pnpxai/explainers/kernel_shap), [LIME](pnpxai/explainers/lime), [IG](pnpxai/explainers/integrated_gradients), [LRP](pnpxai/explainers/lrp), and [RAP](pnpxai/explainers/rap).
- [**Recommender**](pnpxai/recommender): The recommender module offers a recommender system that can suggest the most suitable XAI methods for a given model and dataset, based on the user’s preferences and goals.
- [**Visualizer**](pnpxai/visualizer): The visualizer module enables users to interact with and explore the results of XAI methods in a web-based dashboard, using various charts, graphs, and widgets.

## Installation

To install `pnpxai`, run the following command:

```bash
# Command lines for installation
pip install -e .
```

## Getting Started

This guide explains how to automatically explain your own models and datasets using the provided Python script. The complete code can be found [here](tutorials/auto_explanation.py).

1. **Setup**: The setup involves setting a random seed for reproducibility and defining the device for computation (CPU or GPU). 
    
    ```python
    import torch
    from pnpxai.utils import set_seed
    
    # Set the seed for reproducibility
    set_seed(seed=0)
    
    # Determine the device based on the availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```
    
2. **Initialize Project**: Projects allow you to group and manage multiple explanation experiments. Initialize your project by providing a name. 
    
    ```python
    from pnpxai import Project
    
    # Initialize Project
    project = Project('Test Project')
    ```
    
3. **Create Experiments**: An experiment is an instance for explaining a specific model and dataset. Before creating an experiment, define the model and dataset to be explained.

    **Automatic explainer selection**: The `project.create_auto_experiment` method automatically selects the most applicable explainers and metrics based on the model architecture using `pnpxai.recommender.XaiRecommender`.
        
    ```python
    from torch.utils.data import DataLoader
    from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image
    
    # Load the model and its pre-processing transform for experiment
    model, transform = get_torchvision_model("vit_b_16")
    model = model.to(device)
    
    # Prepare the dataset and dataloader for experiment
    dataset = get_imagenet_dataset(transform, subset_size=25)
    loader = DataLoader(dataset, batch_size=10)
    
    # Define functions to extract input and target from the data loader
    def input_extractor(x): return x[0].to(device)
    def target_extractor(x): return x[1].to(device)
    
    # Define helper functions to visualize input and target
    def input_visualizer(x): return denormalize_image(x, transform.mean, transform.std)
    def target_visualizer(x): return dataset.dataset.idx_to_label(x.item())
    
    # Create an experiment to explain the defined model and dataset
    experiment_vit = project.create_auto_experiment(
        model,
        loader,
        name='ViT Experiment',
        input_extractor=input_extractor,
        target_extractor=target_extractor,
        input_visualizer=input_visualizer,
        target_visualizer=target_visualizer,
    )
    ```
        
    **Manual explainer selection**: Alternatively, you can manually specify the desired explanation method and evaluation metric using `project.create_experiment`.
        
    ```python
    from pnpxai.explainers import LRP
    from pnpxai.evaluator import MuFidelity
    
    experiment_vit = project.create_experiment(
        model,
        loader,
        name='ViT Experiment',
            explainers=[LRP(model)], # desired explanation method
            metrics=[MuFidelity()], # desired evaluation metric
        input_extractor=input_extractor,
        target_extractor=target_extractor,
        input_visualizer=input_visualizer,
        target_visualizer=target_visualizer,
    )
    ```
        
4. **Launch Dashboard**: The web-based dashboard provides information about the model, including model architectures and applicable explanation methods, and visualizes explanation results interactively. Launch the dashboard by running the visualization server from any pre-defined project and entering http://IP_ADDRESS:PORT in your web browser.
    
    ```python
    # Launch the interactive web-based dashboard by running one of the projects defined above
    project.get_server().serve(debug=True, host='0.0.0.0', port=5001)
    ```

Launching the dashboard will display an interactive interface as follows:

<div align='center'>
    <img src="assets/pnpxai_demo.gif" width="100%">
</div>
<div align = "center">
    Example PnP XAI Dashboard
</div>

## License

PnP XAI is released under Apache license 2.0. See [LICENSE](LICENSE) for additional details.
