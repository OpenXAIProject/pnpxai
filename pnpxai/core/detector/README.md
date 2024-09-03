# Detector

`detect_model_architecture` traces a `model: torch.nn.Module`'s graph by using `torch.fx.symbolic_trace` and `model.named_modules()` to extract information to be applied to recommend available explainers. It returns a `Set[torch.nn.Module]` which describes layer informations and representative type of architecture.

## Basic Usage

```python
import torchvision
from pnpxai.detector import detect_model_architecture

model = torchvision.models.get_model("resnet18").eval()
detected_modules = detect_model_architecture(model)
print(detected_modules)
```

## Notes

Since `detect_model_architecture` is fully depending on `torch.fx.symbolic_trace` and `(torch.nn.Module()).named_modules()`, the detected model architecture omits dynamic control flows and non-torch functions in `model.forward`, as [the official document of torch.fx](#https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing) describing. On the contrary, we can say that all operations in static control flow, torch function or python-builtin function are fully detected. If your `model.forward` has dynamic control flows or non-torch functions, please change them to static flows or torch functions in order to let the following procedures of `pnpxai` work in the proper way with your model.