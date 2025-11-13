import numpy as np 
from typing import Any, Callable, Iterable
from math import factorial
from itertools import chain, combinations

class ShapFromScratchExplainer():
    def __init__(self,
                 model: Callable[[np.ndarray], float], 
                 background_dataset: np.ndarray,
                 max_samples: int = None):
        self.model = model
        if max_samples:
            max_samples = min(max_samples, background_dataset.shape[0]) 
            rng = np.random.default_rng()
            self.background_dataset = rng.choice(background_dataset, 
                                                 size=max_samples, 
                                                 replace=False, axis=0)
        else:
            self.background_dataset = background_dataset

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        "SHAP Values for instances in DataFrame or 2D array"
        shap_values = np.empty(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                shap_values[i, j] = self._compute_single_shap_value(j, X[i, :])
        return shap_values
       
    def _compute_single_shap_value(self, 
                                   feature: int,
                                   instance: np.array) -> float:
        "Compute a single SHAP value (equation 4)"
        n_features = len(instance)
        shap_value = 0
        for subset in self._get_all_other_feature_subsets(n_features, feature):
            n_subset = len(subset)
            prediction_without_feature = self._subset_model_approximation(
                subset, 
                instance
            )
            prediction_with_feature = self._subset_model_approximation(
                subset + (feature,), 
                instance
            )
            factor = self._permutation_factor(n_features, n_subset)
            shap_value += factor * (prediction_with_feature - prediction_without_feature)
        return shap_value
    
    def _get_all_subsets(self, items: list) -> Iterable:
        return chain.from_iterable(combinations(items, r) for r in range(len(items)+1))
    
    def _get_all_other_feature_subsets(self, n_features, feature_of_interest):
        all_other_features = [j for j in range(n_features) if j != feature_of_interest]
        return self._get_all_subsets(all_other_features)

    def _permutation_factor(self, n_features, n_subset):
        return (
            factorial(n_subset) 
            * factorial(n_features - n_subset - 1) 
            / factorial(n_features) 
        )
    
    def _subset_model_approximation(self, 
                                    feature_subset: tuple[int, ...], 
                                    instance: np.array) -> float:
        masked_background_dataset = self.background_dataset.copy()
        for j in range(masked_background_dataset.shape[1]):
            if j in feature_subset:
                masked_background_dataset[:, j] = instance[j]
        conditional_expectation_of_model = np.mean(
            self.model(masked_background_dataset)
        )
        return conditional_expectation_of_model