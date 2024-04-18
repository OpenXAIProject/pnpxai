"""Segmentation for time series."""
import stumpy

import numpy as np

from abc import ABC, abstractmethod

from pyts.approximation import SymbolicAggregateApproximation


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


class MatrixProfileSegmentation(AbstractSegmentation):
    """Matrix Profile Segmentation using a matrix profile on every feature."""

    def __init__(self, partitions, win_length=3):
        """Construct segmenter of a time series using matrix profile.

        Args:
            partitions (int): number of partitions
            win_length (int, optional): window length. Defaults to 3.
        """
        self.partitions = partitions
        self.win_length = max(3, win_length)

    def _segment_with_slopes(self, time_series_sample, m=4, k=10, profile='sorted'):
        """Time series instance segmentation into segments.

        Idea:
         - Take the matrix profile of a time series and sort the distances.
         - Calculate the slope of this new matrix profile and take partition largest ones.

        Args:
            time_series_sample (ndarray): Time series data (n_steps, n_features)
            m (int, optional): Windows Size of subsequent to do matrix profile. Defaults to 4.
            k (int, optional): Number of partitions. Defaults to 10.
            profile (str, optional): Sort the corresponding matrix profile before slope or not ('sorted', 'not-sorted').
            Defaults to 'sorted'.

        Returns:
            segmentation_mask: the segmentation mask of a time series. It has the same shape with time series sample.
        """
        # create segmentation mask as the time series
        segmentation_mask = np.zeros_like(time_series_sample)

        # extract steps and features
        n_steps, n_features = time_series_sample.shape

        # set matrix profile window length
        mp_win_len = m
        if mp_win_len == -1:
            # calculate partitions based matrix profile length
            mp_win_len = int(n_steps / k)

        # set first window index to 0
        win_idx = 0
        # create a matrix profile for every feature
        for feature in range(n_features):

            # extract matrix profile with the previously set window length
            mp = stumpy.stump(time_series_sample[:, feature], mp_win_len)
            mp_ = mp[:, 0]  # just take the matrix profile
            temp_mp = mp_
            if profile == 'sorted':
                mp_sorted = sorted(mp_)  # sort values
                temp_mp = mp_sorted
            mp_idx_sorted = np.argsort(mp_)  # sort indeces with values

            # find the largest matrix profile slopes
            # calculate the slopes for every matrix profile step
            slopes = np.array([(temp_mp[i] - temp_mp[i + 1]) / (i - (i + 1))
                              for i in range(len(temp_mp) - 1)])
            # take amount of partitions of the largest slopes
            slopes_sorted = np.argsort(slopes)[::-1][:k]
            # retrieve indeces of original time series
            partitions_idx_sorted = sorted(
                [mp_idx_sorted[part] for part in slopes_sorted])
            # add end of time series
            partitions_idx_sorted.append(n_steps)

            # create windows segmentation masks
            start = 0
            for idx in partitions_idx_sorted:
                end = idx
                segmentation_mask[start:end, feature] = win_idx
                win_idx += 1
                start = idx

            win_idx += 1

        return segmentation_mask

    @staticmethod
    def _segment_with_bins(time_series_sample, m=4, k=10, distance_method='max'):
        """Segment a time series by using matrix profile distance and its bins.

        For shared points between two windows, it can minimize or maximize the nearest distance.

        Args:
            time_series_sample (ndarray): Time series data (n_steps, n_features)
            m (int, optional): Windows Size of subsequent to do matrix profile. Defaults to 4.
            k (int, optional): Initial max number of partitions. Defaults to 10.
                The final result is possiblily smaller than k paritions. Defaults to "max".
            distance_method (str, optional): Options can be `min`, `max`.
                Minimize or maximize the shared points between two windows.

        Returns:
            segmentation_mask: the segmentation mask of a time series.
                It has the same shape with time series sample.
        """
        n_steps, n_features = time_series_sample.shape
        segmentation_mask = np.zeros_like(time_series_sample)
        seg_start = 0
        for feature in range(n_features):
            ts = time_series_sample[:, feature]

            # Get Matrix Profile Distance
            mp = stumpy.stump(ts, m)
            mp_d = mp[:, 0].astype(float)
            mp_d_min = min(0, mp_d.min())
            mp_d_max = mp_d.max()

            # Create bins of distance from min to max
            # segments number
            #   lower: more similar -> motif classes
            #   highest: more dissimilar -> discord classes
            bins = np.linspace(mp_d_min, mp_d_max, k)
            segments = np.digitize(mp_d, bins) - 1  # -1 to make start from 0
            segments = seg_start + segments

            # unpack segments to time series
            #   Notice: For the shared points between two windows, the segment can be maximized, or minimized
            if distance_method == 'max':
                init_v = min(segments)
                _fn = np.fmax
            if distance_method == 'min':
                init_v = max(segments)
                _fn = np.fmin

            seg_m = np.full(n_steps, init_v)
            for i, s in enumerate(segments):
                seg_m[i: i + m] = _fn(seg_m[i: i + m], s)

            segmentation_mask[:, feature] = seg_m
            seg_start = max(seg_m) + 1
        return segmentation_mask

    def segment(self, time_series_sample, segmentation_method='slopes-sorted'):
        """Time series instance segmentation into segments.

        Currently only with slopes but more is planned.

        Args:
            time_series_sample (ndarray): Time series data (n_steps, n_features)
            segmentation_method (str, optional): Segmentation method to be used.
                Defaults to 'slopes-sorted'. Possible: slopes-sorted | slopes-max | bins-min | bins-max

        Returns:
            segmentation_mask: the segmentation mask of a time series.
                It has the same shape with time series sample.
        """
        time_series_sample = time_series_sample.astype(float)
        if segmentation_method == 'slopes-sorted':
            return self._segment_with_slopes(time_series_sample,
                                             m=self.win_length,
                                             k=self.partitions,
                                             profile='sorted')
        if segmentation_method == 'slopes-not-sorted':
            return self._segment_with_slopes(time_series_sample,
                                             m=self.win_length,
                                             k=self.partitions,
                                             profile='not-sorted')

        if segmentation_method == 'bins-max':
            return self._segment_with_bins(time_series_sample,
                                           m=self.win_length,
                                           k=self.partitions,
                                           distance_method='max')
        if segmentation_method == 'bins-min':
            return self._segment_with_bins(time_series_sample,
                                           m=self.win_length,
                                           k=self.partitions,
                                           distance_method='min')

        raise ValueError()


class SAXSegmentation(AbstractSegmentation):
    """SAX Segmentation using a on every feature."""

    def __init__(self, partitions, win_length=3):
        """Construct segmenter for sax algorithm.

        Args:
            partitions (int): number of partitions
        """
        self.partitions = partitions

    def segment(self, time_series_sample, **_kwargs):
        """Time series instance segmentation into segments.

        Idea:
         - Segment data using the SAX transformation.
         - Use SAX to transform data.
         - Use transformed data to identify windows.

        Args:
            time_series_sample (ndarray): Time series data (n_steps, n_features)

        Returns:
            segmentation_mask: the segmentation mask of a time series. It has the same shape with time series sample.
        """
        # create segmentation mask as the time series
        segmentation_mask = np.zeros_like(time_series_sample)

        # set partition amount
        partitions = self.partitions

        # extract steps and features
        n_steps, n_features = time_series_sample.shape

        # set first window index to 0
        win_idx = 0

        # create a sax transformation for every feature
        for feature in range(n_features):

            # check for constant values and use fixed windows for a segmentation
            if np.min(time_series_sample[:, feature]) == np.max(time_series_sample[:, feature]):
                win_len = int(n_steps / partitions)
                for i in range(n_steps):
                    if i % win_len == 0:
                        win_idx += 1
                    segmentation_mask[i, feature] = win_idx

                win_idx += 1
                continue

            # start with a threshold bin size
            n_bins = 3

            internal_win_idx = 0
            while True:
                if not (n_bins < (min(n_steps, 26) - 1)):
                    break
                if not (internal_win_idx < partitions * 8 / 10):
                    if not (internal_win_idx > partitions * 11 / 10):
                        break
                    if not (internal_win_idx < partitions * 14 / 10):
                        break

                # create SAX transformation on the time series feature with the current bin count and
                #   use a quantile partition
                sax = SymbolicAggregateApproximation(
                    n_bins=n_bins, strategy='quantile', alphabet='ordinal')
                sax_transformation = sax.fit_transform(
                    time_series_sample[:, feature].reshape(1, -1))

                internal_win_idx = 0
                old_value = None
                for i, value in enumerate(sax_transformation.reshape(-1)):
                    if old_value and value != old_value:
                        win_idx += 1
                        internal_win_idx += 1
                    segmentation_mask[i, feature] = win_idx
                    old_value = value

                n_bins += 1

            win_idx += 1

        return segmentation_mask


class WindowSegmentation(AbstractSegmentation):
    """Windows segmentation with non-overlapping windows."""

    def __init__(self, partitions, win_length=3):
        """Construct the window segmenter for time series.

        A time series is segmented into uniform or exponential windows.

        Args:
            partitions (int): number of partitions
            win_length (int, optional): windows length. Defaults to 3.
        """
        self.partitions = partitions
        self.win_length = max(3, win_length)

    @staticmethod
    def _segment_with_uniform(time_series_sample, m=4):
        """Segment a time series into uniform windows with the same window size.

        Notice: The window size at the end or begining could be smaller if n_steps % window_lenth != 0
        """
        n_steps, features = time_series_sample.shape
        assert n_steps > m, "Window size must be larger than n-steps"

        starts = list(np.arange(0, n_steps, m))
        ends = starts[1:] + [n_steps]

        segmentation_mask = np.zeros_like(time_series_sample)
        win_idx = 0
        for feature in range(features):
            for i, j in zip(starts, ends):
                segmentation_mask[i: j, feature] = win_idx
                win_idx += 1

        return segmentation_mask

    @staticmethod
    def _segment_with_exponential(time_series_sample):
        """Segment a time series into exponential windows."""
        n_steps, features = time_series_sample.shape

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

        segmentation_mask = np.zeros_like(time_series_sample)
        win_idx = 0
        for feature in range(features):
            for i, j in zip(starts, ends):
                segmentation_mask[i: j, feature] = win_idx
                win_idx += 1

        return segmentation_mask

    def segment(self, time_series_sample, segmentation_method='uniform'):
        """Time series instance segmentation into segments.

        Args:
            time_series_sample (ndarray): Time series data (n_steps, n_features)
            segmentation_method (str, optional): Segmentation method to be used. Defaults to 'uniform'.
                Possible: uniform | exponential

        Returns:
            segmentation_mask: the segmentation mask of a time series. It has the same shape with time series sample.
        """
        time_series_sample = time_series_sample.astype(float)
        if segmentation_method == 'uniform':
            return self._segment_with_uniform(time_series_sample,
                                              m=self.win_length)
        if segmentation_method == 'exponential':
            return self._segment_with_exponential(time_series_sample)


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


class SegmentationPicker:
    """Picker for segmentation method."""

    @staticmethod
    def select(method, partitions, win_length=0):
        """Construct a segmentation method family.

        Args:
            method (str): Segmentation method family, either 'sax', 'window' or 'matrix'.
            partitions (int): number of partitions.
            win_length (int, optional): Window length. Defaults to 0.

        Returns:
            object: An instance of AbstractSegmentation
        """
        if method == 'sax':
            return SAXSegmentation(partitions)
        elif method == 'window':
            return WindowSegmentation(partitions, win_length)
        elif 'matrix' in method:
            return MatrixProfileSegmentation(partitions, win_length)
        else:
            return None
