"""Calculate replacements for a time series."""
import numpy as np
import torch

def _reshape_x(x, segments):
    assert x.shape == segments.shape, (
        f'{x.shape} does not match with segments shape {segments.shape}')
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
        segments = segments.reshape(-1, 1)
    return x, segments


def zeros(x, *_args, **_kwargs):
    """Zeros replacement.

    Args:
        x (ndarray): a time series with any shape.

    Returns:
        ndarray: zeros like x.
    """
    return np.zeros_like(x)


def local_mean(x, segments, *_args, **_kwargs):
    """Local mean replacements for each segments.

    Args:
        x (ndarray): a time series with any shape.
        segments (ndarray): Segmentation of the time series. Same shape with x.
    Returns:
        ndarray: the average per segment per feature in x.
    """
    x, segments = _reshape_x(x, segments)
    _, features = x.shape

    r = np.zeros_like(x).astype(float)
    for i in range(features):
        # Get average per segment
        for s in np.unique(segments):
            idx = (segments[:, i] == s)
            r[idx, i] = np.average(x[idx, i])
    return r


def global_mean(x, *_args, **_kwargs):
    """Global mean replacements for each segments.

    Args:
        x (ndarray): a time series with any shape.
    Returns:
        ndarray: the average of whole x per feature.
    """
    # Consider each feature as one segments (of 1s)
    r = local_mean(x, np.ones_like(x))
    return r


def local_noise(x, segments, *_args, **_kwargs):
    """Local noise for each segments.

    Args:
        x (ndarray): a time series with any shape.
        segments (ndarray): Segmentation of the time series. Same shape with x.
    Returns:
        ndarray: the average per segment per feature in x.
    """
    pass


def global_noise(x, *_args, **_kwargs):
    """Global noise for each segments.

    Args:
        x (ndarray): a time series with any shape.
    Returns:
        ndarray: the average per feature in x.
    """
    pass


def inverse_max(x, *_args, **_kwargs):
    """Inverse max replacement.

    Args:
        x (ndarray): a time series with any shape.
    Returns:
        ndarray: the average per segment per feature in x.
    """
    _, n_features = x.shape
    max_v = x.max(axis=0).reshape(-1, n_features)
    r = max_v - x
    return r


def inverse_mean(x, *_args, **_kwargs):
    """Inverse mean replacement.

    Args:
        x (ndarray): a time series with any shape.
    Returns:
        ndarray: the inversed average per feature in x.
    """
    mean_v = global_mean(x)
    r = mean_v - x
    return r


def random(x, *_args, **_kwargs):
    """Random replacements.

    Args:
        x (ndarray): a time series with any shape.
    Returns:
        ndarray: random array of x.
    """
    r = np.random.rand(*x.shape)
    return r
