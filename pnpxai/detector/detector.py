from dataclasses import dataclass
from ._core import ModelArchitecture

@dataclass
class DetectorOutput:
    architecture: set


class ModelArchitectureDetector:
    """
    Detects the architecture of a model by analyzing its modules.

    Attributes:
    - modules: A list to store detected modules.
    """
    def __init__(self):
        self.modules = []
    
    def extract_modules(self, model):
        """
        Extracts modules from the model.

        Args:
        - model: The model from which to extract modules.
        """
        module_mode = model.training
        model.eval()

        ma = ModelArchitecture(model)
        for n in ma.list_nodes():
            if n.opcode == "call_module":
                self.modules.append(n.operator)
        model.training = module_mode

    def __call__(self, model):
        """
        Calls the ModelArchitectureDetector object.

        Args:
        - model: The model to analyze.

        Returns:
        - DetectorOutput: An object containing the detected architecture.
        """
        self.extract_modules(model)
        return DetectorOutput(
            architecture = set([type(module) for module in self.modules])
        )