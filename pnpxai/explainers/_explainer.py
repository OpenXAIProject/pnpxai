from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, List, Dict

from pnpxai.core._types import Model, DataSource
from .utils.feature_mask import get_default_feature_mask

class Explainer:
    def __init__(
            self,
            source: Any,
            model: Model,
        ):
        self.source = source
        self.model = model

        self.additional_kwargs = self.get_default_additional_kwargs()
        self.device = next(self.model.parameters()).device

    @property
    @abstractmethod
    def _attributor_arg_keys(self) -> List[str]:
        return []

    @abstractmethod
    def get_default_additional_kwargs(self) -> Dict:
        return {}
    
    def attribute(
        self,
        inputs: DataSource,
        targets: DataSource,
        **kwargs,
    ) -> DataSource:
        current_additional_kwargs = self.additional_kwargs.copy()
        if "feature_mask" in current_additional_kwargs:
            current_additional_kwargs["feature_mask"] = get_default_feature_mask(inputs, self.device)
        for k, v in kwargs.items():
            current_additional_kwargs[k] = v

        # select kwargs for attributor
        attributor_kwargs = {}
        for k in self._attributor_arg_keys:
            attributor_kwargs[k] = current_additional_kwargs.pop(k)
        attributor = self.source(self.model, **attributor_kwargs)

        # attribute
        attr = attributor.attribute(inputs, target=targets, **current_additional_kwargs)
        return attr
    