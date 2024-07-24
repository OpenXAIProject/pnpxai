from typing import Sequence
import pandas as pd
import numpy as np

def format_into_array(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy()
    if isinstance(obj, pd.Series):
        return obj.to_numpy()
    if isinstance(obj, Sequence) and not isinstance(obj, str):
        return np.array(obj)
    return obj


def iterate_inputs(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.iterrows()
    return enumerate(obj)