from typing import Optional, Callable

import torch
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.linear_model._base import LinearModel
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

from pnpxai.core._types import Model, DataSource
from pnpxai.explainers_backup._explainer import Explainer
from pnpxai.explainers_backup.ts_mule.sampling.perturb import Perturbation, AbstractPerturbation
from pnpxai.explainers_backup.ts_mule.sampling.segment import UniformSegmentation, AbstractSegmentation


class TSMule(Explainer):
    def __init__(
        self,
        model: Model,
        kernel: Optional[LinearModel] = None,
        sampler: Optional[AbstractPerturbation] = None,
        segmenter: Optional[AbstractSegmentation] = None,
        n_samples: int = 100,
        prob_disable: float = 0.5,
        replace_method: Optional[Callable] = None,
    ) -> None:
        super(TSMule, self).__init__(model)
        self._kernel: LinearModel = kernel if kernel is not None else Lasso(
            alpha=.01)
        self._sampler: AbstractPerturbation = sampler if sampler is not None else Perturbation(
            n_samples=n_samples,
            prob_disable=prob_disable,
            replace_method=replace_method
        )
        self._segmenter: AbstractSegmentation = segmenter if segmenter is not None else UniformSegmentation()

        self._coef = None
        self._xcoef = None

    def _fit_kernel(self, z_hat, z_prime, pi):
        # Try to approximate g(z') ~ f(new_x) <=> g(z') = Z'* W ~ Z_hat
        _t = train_test_split(
            z_prime,
            z_hat,
            pi,
            test_size=0.3
        )
        X, X_test, y, y_test, sw, sw_test = _t

        # Avoid nan in similarity
        sw = np.nan_to_num(np.abs(sw), 0.01)
        sw_test = np.nan_to_num(np.abs(sw_test), 0.01)

        # Fit g(z') ~ f(new_x)
        self._kernel.fit(X, y, sample_weight=sw)

        # Evaluation Score
        y_pred = self._kernel.predict(X_test)
        y_pred = self._kernel.predict(X_test)
        score = metrics.r2_score(y_test, y_pred)

        return score

    def _predict(self, data: np.ndarray, seq_dim: int) -> np.ndarray:
        model_device = next(self.model.parameters()).device
        data = torch.Tensor(data).to(model_device)

        data = data.transpose(-2, seq_dim)
        outputs = self.model(data).cpu().detach()

        return outputs

    def attribute(
        self,
        inputs: DataSource,
        targets: DataSource,
        seq_dim: int = -1,
        *args,
        **kwargs
    ):
        assert inputs.ndim == 3, "Input data smust be 3D (Batch, Sequence, Features)"
        device = inputs.device
        inputs = inputs.cpu().detach().numpy()
        inputs = inputs.swapaxes(-2, seq_dim)
        # Get segmentation masks

        # Generate samples
        coefs = []
        for cur_inputs, cur_target in zip(inputs, targets):
            segment_mask = self._segmenter.segment(cur_inputs)
            new_x, z_prime, pi = self._sampler.perturb(
                cur_inputs, segment_mask)

            outputs = self._predict(new_x, seq_dim)
            self._fit_kernel(outputs, z_prime, pi)

            # Set coef of segments
            coef = np.array(self._kernel.coef_[cur_target])
            coef = self.to_original(coef.ravel(), segment_mask)
            coefs.append(coef)

        coefs = torch.Tensor(coefs).to(device)
        coefs = coefs.transpose(-2, seq_dim)

        return coefs

    @staticmethod
    def to_original(coef, segments):
        """Convert coef per segment to coef per point.

        Args:
            coef (array): Coefficients of unique segments.
            segments (ndarray): Original segmentations of its time series.

        Returns:
            ndarray: coefficients of each point and have same shape with the time series (n_steps, n_features).
        """
        x_coef = np.zeros_like(segments).astype(float)

        # Get labels vectors from segmentation
        seg_unique_labels = np.unique(segments)
        assert coef.shape == seg_unique_labels.shape

        for i, l in enumerate(seg_unique_labels):
            idx = (segments == l)
            x_coef[idx] = coef[i]
        return x_coef
