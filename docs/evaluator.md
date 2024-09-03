# Evaluator <small>[[source](api/evaluator/metrics.md)]</small>

This module combines multiple evalulation metrics:

* [**Complexity**](api/evaluator/metrics/#pnpxai.evaluator.metrics.complexity.Complexity): Complexity metric
* [**MuFidelity**](api/evaluator/metrics/#pnpxai.evaluator.metrics.complexity.MuFidelity): MuFidelity metric
* [**Sensitivity**](api/evaluator/metrics/#pnpxai.evaluator.metrics.complexity.Sensitivity): Sensitivity metric
* [**Area between Perturbation Curves**](api/evaluator#pnpxai.evaluator.metrics.complexity.AbPC): AbPC metric

Each metric is designed to address specific features of an explainer. Therefore, final ranking of the relative quality of explainers' is computed by combining several metrics.