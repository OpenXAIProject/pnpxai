import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from abc import ABC, abstractmethod
from typing import Optional, Callable

from pnpxai.explainers.ts_mule.sampling.replace import zeros


class AbstractPerturbation(ABC):
    """Abstract Pertubation with abstract methods."""

    @abstractmethod
    def __init__(self, **kwargs):
        """Abstract construction."""
        self.p_off = None
        self.repl_method = None
        self.n_samples = None

    @abstractmethod
    def perturb(self, ts, segments):
        """Perturb a time series to create new sample with same shape.

        :param ts: (np.array) A time series must be (n_steps, n_features)
        :param segments: (np.array) A segments with labels of the time series must be (n_steps, n_features)

        Yields:
            Generator: tuple of (new sample, on/off segments, similarity to original)
        """
        pass


class Perturbation(AbstractPerturbation):
    """Base Perturbation module."""

    def __init__(
        self,
        n_samples: int = 100,
        prob_disable: float = 0.5,
        replace_method: Optional[Callable] = None,
    ):
        """Construct perturbation base module.

        Args:
            p_off (float, optional): Probability of disabling a segment. Default is 0.5
            method (str, optional): Methods to replace parts of segmentation, including:
                'zeros | global_mean | local_mean | inverse_mean | inverse_max'
                Defaults to 'zeros'.
            n_samples (int, optional): [description]. Defaults to 10.
        """
        self.prob_disable = prob_disable
        self.replace_method = replace_method if replace_method is not None else zeros
        self.n_samples = n_samples

    def _get_on_off_segments(self, segments: np.ndarray) -> np.ndarray:
        # Get n_seg
        n_seg = len(np.unique(segments))

        # Get on off segments
        # 0 = off/disabled/replaced, 1 = on/keep/unchanged
        probs = np.random.choice(
            [0, 1],
            size=n_seg,
            p=[self.prob_disable, 1 - self.prob_disable]
        )
        return probs

    def _get_segment_mask(self, segments: np.ndarray, on_off_segments: np.ndarray) -> np.ndarray:
        # Get binary on/off masks for segments
        labels = np.unique(segments)
        n_segments = len(labels)
        mask = np.ones_like(segments)
        mask = sum([
            np.where(segments == labels[i], on_off_segments[i], mask)
            for i in range(n_segments)
        ])
        return mask

    def _get_similarity(self, x: np.ndarray, z: np.ndarray, method: Optional[Callable] = None) -> float:
        # Calculate pi/similarity between x and y:
        pi = 1.
        method = method if method is not None else kendalltau
        if method in {pearsonr, spearmanr, kendalltau}:
            pi, _ = method(x.ravel(), z.ravel())
            pi = np.nan_to_num(pi, 0.01)

        return pi

    def get_sample(self, inputs: np.ndarray, segments: np.ndarray, replaced: np.ndarray = None):
        """Get sample of x based on replace segments of x with r.

        Args:
            x (ndarray): A multivariate time series
            segm (ndarray): A segmentation of x, having same shape with x
            r (ndarray): A replacements of x when create a new sample
            p_off (float, optional): Probility of disabling a segmentation. Defaults to 0.5.
        Yields:
            Generator: a tuple of (new sample, on/off segments, similarity to original)
        """
        if replaced is None:
            replaced = np.zeros_like(inputs)
        assert replaced.shape == inputs.shape == segments.shape

        # On/off vector z', used to fit into XAI linear regression
        z_prime = self._get_on_off_segments(segments)
        mask = self._get_segment_mask(segments, z_prime)

        # get new x sample, when mask = 1, then keep x, else replace it
        new_x = inputs * mask + replaced * (1 - mask)
        pi = self._get_similarity(inputs, new_x)
        return new_x, z_prime, pi

    def get_samples(self, x: np.ndarray, segments: np.ndarray):
        """Perturb and generate sample sets from given time series and its segmentation.

        Args:
            ts (np.ndarray): A time series with shape (n_steps, n_features)
            segments (np.ndarray): A segmentation of the time series with shape (n_steps, n_features)
            replace_method (str): Method to replace off/disabled segment
            p_off (float): Probability of disabling a segment. Default is 0.5
            n_samples (int): Number of samples to be generated.

        Yields:
            Generator: tuples of (new sample, on/off segments, similarity to original)
        """
        r = self.replace_method(x, segments)

        return list(zip(*[
            self.get_sample(x, segments, r)
            for _ in range(self.n_samples)
        ]))

    def perturb(self, ts: np.ndarray, segments: np.ndarray):
        """Perturb and generate sample sets from given time series and its segmentation.

        Args:
            ts (np.ndarray): A time series with shape (n_steps, n_features)
            segments (np.ndarray): A segmentation of the time series with shape (n_steps, n_features)

        Yields:
            Generator: tuple of (new sample, on/off segments, similarity to original)
        """
        return self.get_samples(ts, segments)
