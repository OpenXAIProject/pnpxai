from dataclasses import dataclass
from ._core import ModelArchitecture

@dataclass
class DetectorOutput:
    architecture: set


class ModelArchitectureDetector:
    def __init__(self):
        self.modules = []
    
    def extract_modules(self, model):
        module_mode = model.training
        model.eval()

        ma = ModelArchitecture(model)
        for n in ma.list_nodes():
            if n.opcode == "call_module":
                self.modules.append(n.operator)
        model.training = module_mode

    def __call__(self, model):
        self.extract_modules(model)
        return DetectorOutput(
            architecture = set([type(module) for module in self.modules])
        )