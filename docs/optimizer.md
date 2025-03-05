# Optimizer <small>[[source](api/evaluator/optimizer.md)]</small>

This module is designed to optimize experiment hyperparameters. The optimization procedure can be invoked by ```python experiment.optimize(...)``` method call.

Optimization procedure is based on [`optuna`](https://optuna.org/) package.

The optimization method requires initialization of an `Experiment`, where `modality`, model, explainer, and metrics are specified. To leverage chaching procedure, defined in experiment, the method invocation requires ids of the aforementioned parameters to be passed instead of instances themselves. Therefore, the method call is as follows:

Complete backbone of initialization code can be represented as follows:

```python
import torch
from torcu.utils.data import DataLoader

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
experiment.optimize(
    data_ids=0,
    explainer_id=0,
    metric_id=0,
    direction='maximize',
    sampler='random',
    n_trials=50,
    seed=42,
)
```

When manually defining custom explainers, it is necessary to implement `get_tunables` method to enable optimization procedure in the `Experiment` module. This method is expected to provide a dictionary with hyperparameter names as keys, and options as values. An example of a custom explainer is presented below:

```python
class CustomExplainer(Explainer):
    SUPPORTED_MODULES = [...]

    def __init__(
        self,
        model: Module,
    ) -> None:
        super(CustomExplainer, self).__init__(model)
        self.hp1 = ...
        self.hp2 = ...

        
    def attribute(
        self,
        inputs: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor]]:
        attrs = ...
        return attrs

    def get_tunables(self) -> Dict[str, Tuple[type, Dict]]:
        return {
            'hp1': (int, {'low': 10, 'high': 50, 'step': 10}),
            'hp2': (BaselineFunction, {}),
        }
```

Here `get_tunables` method suggests an optimization module to optimize the `hp1`, and `hp2` attributes following `suggest_int` method, defined in `optuna` package, and `BaselineFunction`, specified by `Modality.map_fn_selector`.
