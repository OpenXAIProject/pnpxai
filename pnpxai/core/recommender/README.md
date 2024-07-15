# Recommender

The Recommender module in `pnpxai` assists you in selecting the most suitable explanation method and evaluation metric for your specific Explainable AI (XAI) needs. It considers various factors like question types, tasks, neural network architectures, and your desired evaluation criteria.

## Key Features

- **Automated Method Selection:** Leverages the provided table to intelligently recommend applicable methods based on your inputs.
- **Customizable:** Offers flexibility to tailor the selection process by refining the table or implementing your own filters.
- **Comprehensive Coverage:** Supports a wide range of explanation methods and metrics for diverse XAI use cases.
- **Easy Integration:** Seamlessly integrates into your `pnpxai` workflows for streamlined explainer development.

## Supported Methods and Metrics

The following table summarizes the methods and metrics currently supported by the Recommender module.

| Method | Questions | Tasks | Architectures | Metrics |
| --- | --- | --- | --- | --- |
| Grad-CAM | Why | Image | Convolution | Correctness, Continuity |
| LIME | Why | Image, Tabular Data, Text | Linear, Convolution, Recurrent, Transformer, Decision Trees | Correctness, Continuity |
| SHAP | Why | Image, Tabular Data, Text | Linear, Convolution, Recurrent, Transformer | Correctness, Continuity |
| IG | Why | Image, Text | Linear, Convolution, Recurrent, Transformer | Correctness, Continuity |
| LRP | Why | Image, Text | Linear, Convolution, Recurrent, Transformer | Correctness, Continuity |
| PDP | How | Tabular Data | Decision Trees | None |
| CEM | Why not | Image, Tabular Data, Text | Linear, Convolution, Recurrent, Transformer | Completeness |
| TCAV | Why | Image | Linear, Convolution, Recurrent, Transformer | Correctness |
| Anchors | Why, How to still be this | Tabular Data | Linear, Convolution, Recurrent, Transformer, Decision Trees | Completeness, Compactness |

## Usage

1. **Import:** Begin by importing the `XaiRecommender` module in your Python code:
    
    ```python
    from pnpxai.recommender import XaiRecommender
    ```
    
2. **Create an Instance:** Construct a `XaiRecommender` object:
    
    ```python
    recommender = XaiRecommender()
    ```
    
3. **Get Recommendations:** Retrieve recommended methods and metrics by calling `XaiRecommender` object with your desired parameters:
    
    ```python
    recommender_output = recommender(question, task, model_architecture)
    print("Recommended methods: ", recommender_output.explainers)
    print("Recommended metrics: ", recommender_output.evaluation_metrics)
    ```
    

## Customization

You can customize the recommendation process by modifying the provided table to reflect your specific requirements or domain knowledge.

The table consists of four Python dictionaries:

- `QUESTION_TO_EXPLAINERS`: Maps question types ("why", "how", etc.) to supported explanation methods.
- `TASK_TO_EXPLAINERS`: Maps task types (e.g., "image", "tabular") to compatible methods.
- `ARCHITECTURE_TO_EXPLAINERS`: Links neural network architectures (e.g., "cnn", "transformer") to suitable explainers.
- `EXPLAINER_TO_METRICS`: Associates explanation methods with relevant evaluation metrics (e.g., "Sensitivity", "Complexity").

By carefully revising these dictionaries, you can guide the Recommender to prioritize methods that align with your specific use case and evaluation criteria.