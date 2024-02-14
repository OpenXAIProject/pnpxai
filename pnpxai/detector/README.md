# Detector

`ModelArchitectureDetector` traces a `model: nn.Module`'s graph by using `torch.fx.symbolic_trace` and its output `traced_model: torch.fx.GraphModule` and extracts information to be applied to recommend available explainers from `node: torch.fx.Node`. In current version, the detection is implemented by `__call__(self, model: nn.Module)` and returns an instance of `DetectorOutput`, which is a dataclass attributed by `architecture`, or a set of module types that make up the model.

## Basic Usage

```python
import torchvision
from pnpxai.detector import ModelArchitectureDetector

model = torchvision.models.get_model("resnet18").eval()
detector = ModelArchitectureDetector()
model_architecture = detector(model)
```

## Notes

Since `ModelArchitectureDetector` is fully depending on `torch.fx.symbolic_trace`, the detected model architecture omits dynamic control flows and non-torch functions in `model.forward`, as [the official document of torch.fx](#https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing) describing. On the contrary, we can say that all operations in static control flow, torch function or python-builtin function are fully detected. If your `model.forward` has dynamic control flows or non-torch functions, please change them to static flows or torch functions in order to let the following procedures of `pnpxai` work in the proper way with your model.