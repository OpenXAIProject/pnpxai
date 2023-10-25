# Evaluator

* 위에 제시된 모든 Explainer들에 대해 아래의 지표 계산
  * Correctness - Infidelity
  * Robustness - Sensitivity
  * Completeness - TBD



```python
import numpy as np

# Example explanation function for demonstration purposes
def phi(f, x):
    return x * f(x)

# Example predictor function
def f(x):
    return np.sum(x**2)
    
    
def infidelity(f, phi, x, I_samples): 
    """ Compute the infidelity measure.
    Parameters:
    - f: Black-box predictor function.
    - phi: Feature attribution explanation function.
    - x: Test input.
    - I_samples: Samples of perturbations.
    
    Returns:
    - infidelity score.
    """
    errors = []
    for I in I_samples:
        term1 = np.dot(I, phi(f, x))
        term2 = f(x) - f(x - I)
        errors.append((term1 - term2) ** 2)
    return np.mean(errors)


def gradient_sensitivity(phi, f, x, epsilon=1e-5):
    """ Compute the gradient sensitivity.

    Parameters:
    - phi: Explanation function.
    - f: Black-box predictor function.
    - x: Test input.
    - epsilon: Small perturbation.

    Returns:
    - Gradient sensitivity score.
    """
    sensitivity = []
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = epsilon
        sensitivity_i = (phi(f, x + e) - phi(f, x)) / epsilon
        sensitivity.append(sensitivity_i)
    return np.max(np.abs(sensitivity))


def local_lipschitz(phi, x, r):
    """ Compute the local Lipschitz continuity.

    Parameters:
    - phi: Explanation function.
    - x: Test input.
    - r: Radius of perturbation.

    Returns:
    - Local Lipschitz continuity score.
    """
    # This is a simplified method using random sampling.
    # It's recommended to use more structured methods for a precise measure.
    num_samples = 100
    max_diff = 0
    for _ in range(num_samples):
        perturbation = np.random.uniform(-r, r, size=x.shape)
        diff = np.linalg.norm(phi(x + perturbation) - phi(x))
        max_diff = max(max_diff, diff)
    return max_diff / r


def max_sensitivity(phi, f, x, r):
    """ Compute the max-sensitivity.

    Parameters:
    - phi: Explanation function.
    - f: Black-box predictor function.
    - x: Test input.
    - r: Radius of perturbation.

    Returns:
    - Max-sensitivity score.
    """
    # This is again a simplified method using random sampling.
    num_samples = 100
    max_diff = 0
    for _ in range(num_samples):
        perturbation = np.random.uniform(-r, r, size=x.shape)
        diff = np.abs(phi(f, x + perturbation) - phi(f, x))
        max_diff = max(max_diff, diff)
    return max_diff
```

* Aggregation
  * Score는 Normalization이 불가능하기 때문에 모든 score의 rank의 평균이 작은 순서대로 plotting
  * 동점이면 infidelity가 가장 큰 순서대로 보여주기
