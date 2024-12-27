import torch
from transformers import AutoModel
from pnpxai.explainers.base import Explainer


class LEAR(Explainer):
    """
    LEAR Explainer class for generating counterfactual explanations.

    Args:
        model_name (str): The name of the pre-trained model to load.
        device (str, optional): Device to use for computation ('cuda' or 'cpu').
        model (torch.nn.Module, optional): This is not used for LEAR. It is included for compatibility with the base class.
    """

    def __init__(self, model=None, model_name: str="wltjr1007/LEAR", device: torch.device=None):
        super().__init__(model=model)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Load pre-trained model for counterfactual generation (trust_remote_code=True allows custom code from model repository)
        self.CMG = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.num_labels = self.CMG.config.num_labels
        self.CMG.eval()

    def attribute(self, inputs, targets):
        """
        Generate counterfactual images by attributing changes to input features.

        Args:
            inputs (torch.Tensor): Input tensor containing original images.
            targets (torch.Tensor): Labels tensor specifying target classes for counterfactuals.

        Returns:
            torch.Tensor: Counterfactual images generated by the model.
        """
        inputs = inputs.to(self.device)

        if targets.dim() == 1:
            targets = torch.nn.functional.one_hot(targets, num_classes=self.num_labels)
        targets = targets.to(self.device)

        with torch.inference_mode():
            counterfactual = self.CMG(inputs, targets)
        return counterfactual