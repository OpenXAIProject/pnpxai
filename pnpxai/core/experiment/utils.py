from typing import Sequence, Any
import re

def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def _format_to_tuple(obj: Any):
    if isinstance(obj, Sequence):
        return tuple(obj)
    return (obj,)