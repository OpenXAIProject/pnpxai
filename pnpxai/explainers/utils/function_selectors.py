from typing import Callable, Optional, Dict, Any


class FunctionSelector:
    """
    A utility class for managing and selecting functions based on keys. This class allows you 
    to define a set of functions associated with specific keys, and provides methods to add, 
    retrieve, and select functions with optional parameters.

    Parameters:
        data (Optional[Dict[str, Callable]], optional):
            A dictionary where the keys are function names (or identifiers) and the values are 
            the corresponding function objects. Defaults to an empty dictionary if not provided.
        default_kwargs (Optional[Dict[str, Any]], optional):
            A dictionary of default keyword arguments to be used when selecting functions. Defaults 
            to an empty dictionary if not provided.

    Attributes:
        choices (List[str]):
            A list of all available function keys.
    
    Methods:
        add(key: str, value: Callable) -> Callable:
            Adds a new function to the selector with the specified key.

        get(key: str) -> Callable:
            Retrieves the function associated with the specified key.

        delete(key: str) -> Optional[Callable]:
            Removes the function associated with the specified key and returns it. If the key does 
            not exist, returns None.

        all() -> List[Callable]:
            Returns a list of all functions currently stored in the selector.

        select(key: str, **kwargs) -> Callable:
            Selects and returns the function associated with the specified key, with the given 
            keyword arguments. Uses default keyword arguments if not overridden.

        get_tunables() -> Dict[str, Any]:
            Returns a dictionary containing tunable parameters for the selector. In this case, it 
            provides the choices of function keys.

    Notes:
        - Ensure that functions added to the selector match the expected signature, as incorrect 
          arguments can lead to runtime errors.
    """
    
    def __init__(
        self,
        data: Optional[Dict[str, Callable]] = None,
        default_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._data = data or {}
        self._default_kwargs = default_kwargs or {}

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
        kwargs = {**self._default_kwargs, **kwargs}
        return fn_type(**kwargs)

    def get_tunables(self):
        return {'method': (list, {'choices': self.choices})}
