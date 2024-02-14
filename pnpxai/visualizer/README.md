# Web Application

The PnP XAI Web Application serves as a powerful interface for applying explainable AI algorithms with ease. It is designed to facilitate users in understanding and interpreting their machine learning models through a comprehensive suite of XAI tools.

## Getting Started

To initiate the local server for the web application, follow these steps:

```
cd tutorials && python demo_tutorial.py
```


This command sets up the server, allowing access through a specified local URL.

## Features

### Experiment Information Page

On the Experiment Information Page, users have the capability to confirm the accuracy of their model detections using our sophisticated algorithms. Moreover, this page provides an option to visualize the model's architecture, offering insights into its operational framework.

![Homepage](assets/experiment_info.png)

### Explainer Results

The cornerstone of our web application is its ability to generate and display a variety of explainer results. Users equipped with their own data and models can utilize this feature to obtain detailed evaluation outcomes. This process aids in the comprehension of the model's decision-making processes and identifies areas for improvement.

![Explainer View](assets/local_explain.png)

## Caveats and Recommendations

- **GPU Memory Limitations**: Insufficient GPU RAM may lead to errors during the computation of evaluation metrics. Users are advised to ensure that their system meets the necessary requirements to prevent such issues.
- **Processing Time**: The duration required to run the code and obtain results may vary based on the complexity of the input data and the model itself. Users should anticipate some waiting time for the processes to complete.

The Web Application segment of our package is instrumental in bridging the gap between complex AI models and actionable insights. By providing an intuitive platform for applying explainable AI algorithms, we aim to enhance the accessibility and understanding of AI systems.