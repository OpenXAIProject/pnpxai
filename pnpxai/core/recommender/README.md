# Recommender

The Recommender module in `pnpxai` assists you in selecting the most suitable explanation methods for your specific Explainable AI (XAI) needs. It considers factors like modality and neural network architecture.

By analyzing all the available explainers, the recommender module extracts the ones fitting the best to user-specified modality and a model.

## Key Features

- **Automated Method Selection:** Leverages the provided table to intelligently recommend applicable methods based on your inputs.
- **Comprehensive Coverage:** Supports a wide range of explanation methods and metrics for diverse XAI use cases.
- **Easy Integration:** Seamlessly integrates into your `pnpxai` workflows for streamlined explainer development.

## Supported Methods and Metrics

The following table summarizes the methods and metrics currently supported by the Recommender module.

| Method | Data Modalities | Architectures |
| --- | --- | --- |
|LIME                   | V, L, SD, TS | Linear, Convolution, Recurrent, Transformer, Decision Trees|
|KernelSHAP             | V, L, SD, TS | Linear, Convolution, Recurrent, Transformer, Decision Trees|
|Gradient               | V, L, TS | Linear, Convolution, Recurrent, Transformer|
|Gradient &times; Input | V, L, TS | Linear, Convolution, Recurrent, Transformer|
|Grad-CAM               | V, TS | Convolution|
|Guided Grad-CAM        | V, TS | Convolution|
|FullGrad               | V | Linear, Convolution, Recurrent, Transformer|
|SmoothGrad             | V, L, TS | Linear, Convolution, Recurrent, Transformer|
|VarGrad                | V, L, TS | Linear, Convolution, Recurrent, Transformer|
|IG                     | V, L, TS | Linear, Convolution, Recurrent, Transformer|
|LRP                    | V, L, TS | Linear, Convolution, Recurrent, Transformer|
|RAP                    | V, L, TS | Linear, Convolution, Recurrent, Transformer|
|AttentionRollout       | V, L | Transformer|
|TransformerAttribution | V, L | Transformer|

<small>
* Supported data modalities are: Vision (V), Language (L), Structured Data (SD), and Time Series (TS)
</small>

## Usage

1. **Import:** Begin by importing the `XaiRecommender` module in your Python code:
    
    ```python
    from pnpxai.core import XaiRecommender
    from pnpxai.core.modality import ImageModality
    ```
    
2. **Create an Instance:** Construct a `XaiRecommender` object:
    
    ```python
    model = ...
    recommender = XaiRecommender()
    modality = ImageModality()
    ```
    
3. **Get Recommendations:** Retrieve recommended methods and metrics by calling `XaiRecommender` object with your desired parameters:
    
    ```python
    recommender_output = recommender(modality, model)
    print("Recommended Explainers: ", recommender_output.explainers)
    print("Detected Architectures: ", recommender_output.detected_architectures)
    ```
