from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class LayerInfo:
    name: str
    module_name: str
    class_name: str
    output_size: Tuple
    grad_fn: Optional[str]

    def type(self):
        return ".".join([self.module_name, self.class_name])


class ModelArchitecture:
    def __init__(self, data: Optional[List[LayerInfo]]=None):
        self.data = data if data else []
    
    def _add_data(self, data: Dict):
        self.data.append(LayerInfo(**data))
