from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Dict


class FunctionSelector:
    def __init__(self, data: Optional[Dict[str, Callable]]=None):
        super().__init__()
        self._data = {} or data

    @property
    def choices(self):
        return list(self._data.keys())

    def add(self, key: str, value: Callable):
        self._data[key] = value
        return value

    def get(self, key: str):
        return self._data[key]

    def delete(self, key: str):
        return self._data.pop(key, None)

    def all(self):
        return [self.get(key) for key in self.choices]

    def select(self, key: str, **kwargs):
        fn_type = self.get(key)
        return fn_type(**kwargs)

    def get_tunables(self):
        return {'method': (list, {'choices': self.choices})}
