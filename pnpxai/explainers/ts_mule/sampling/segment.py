import numpy as np
from abc import ABC, abstractmethod


class AbstractSegmentation(ABC):
    """Abstract Segmentation with abstract methods."""

    @abstractmethod
    def __init__(self):
        """Abstract construct."""
        pass

    @abstractmethod
    def segment(self, time_series_sample):
        """Time series instance segmentation into segments.

        Args:
            time_series_sample (ndarray): Time series data (n_steps, n_features)

        Returns:
            segmentation_mask: the segmentation mask of a time series.
                It has the same shape with time series sample.
        """
        pass


class ExponentialSegmentation(AbstractSegmentation):
    def segment(self, data: np.ndarray):
        """Segment a time series into exponential windows."""
        n_steps, features = data.shape

        # Get possible x (as window size) from exponential(x). Here try to make half of size
        x = np.arange(np.ceil(np.log(n_steps)))

        win_sizes = [np.ceil(np.exp(i)).astype(int) for i in x]

        # Adjust to have total of win_sizes must be equal = n_steps
        win_sizes[-1] = n_steps - sum(win_sizes[:-1])

        # Get starts and ends for each window
        starts = [0]
        for i in range(len(win_sizes) - 1):
            idx = starts[-1] + win_sizes[i]
            starts.append(idx)
        ends = starts[1:] + [n_steps]

        segmentation_mask = np.zeros_like(data)
        win_idx = 0
        for feature in range(features):
            for i, j in zip(starts, ends):
                segmentation_mask[i: j, feature] = win_idx
                win_idx += 1

        return segmentation_mask


class UniformSegmentation(AbstractSegmentation):
    """Windows segmentation with non-overlapping windows."""

    def __init__(self, window_len=3):
        """Construct the window segmenter for time series.

        A time series is segmented into uniform or exponential windows.

        Args:
            partitions (int): number of partitions
            win_length (int, optional): windows length. Defaults to 3.
        """
        self.window_len = window_len

    def segment(self, inputs: np.ndarray):
        """Segment a time series into uniform windows with the same window size.

        Notice: The window size at the end or begining could be smaller if n_steps % window_lenth != 0
        """
        n_steps, features = inputs.shape
        assert n_steps > self.window_len, "Window size must be larger than n-steps"

        starts = list(np.arange(0, n_steps, self.window_len))
        ends = starts[1:] + [n_steps]

        segmentation_mask = np.zeros_like(inputs)
        win_idx = 0
        for feature in range(features):
            for i, j in zip(starts, ends):
                segmentation_mask[i: j, feature] = win_idx
                win_idx += 1

        return segmentation_mask
