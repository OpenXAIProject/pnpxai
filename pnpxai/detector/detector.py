from dataclasses import dataclass
from functools import partial
import torch.nn as nn


@dataclass
class DetectorOutput:
    architecture: set


class ModelArchitectureDetector:
    def __init__(self):
        self.modules = []

    def extract_modules(self, model):
        modules = []
        for module in model.children():
            if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
                modules.extend(self.extract_modules(module))
            else:
                modules.append(module)
        return modules

    def __call__(self, model):
        self.modules = self.extract_modules(model)
        return DetectorOutput(
            architecture = set([type(module) for module in self.modules])
        )


class ModelArchitectureDetectorV2:
    def __init__(self):
        self.modules = []
        # self.inputs = []
        # self.outputs = []

    def _record_layer_info(self, module, input, output):
        self.modules.append(module)
        # self.inputs.append(input)
        # self.outputs.append(output)
    
    def extract_modules(self, model, sample):
        module_mode = model.training
        model.eval()

        hook_handles = []
        for name, module in model.named_modules():
            hook_handles.append(module.register_forward_hook(partial(self._record_layer_info)))
        model(sample)
        for handle in hook_handles:
            handle.remove()
        model.training = module_mode

    def __call__(self, model, sample):
        self.extract_modules(model, sample)
        # pdb.set_trace()
        return DetectorOutput(
            architecture = set([type(module) for module in self.modules])
        )