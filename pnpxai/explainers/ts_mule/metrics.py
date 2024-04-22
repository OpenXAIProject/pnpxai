from torch import Tensor


def r2_score(y_true: Tensor, y_pred: Tensor):
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - numerator / denominator


def kendall_tau(x: Tensor, y: Tensor, dim: int = -1):
    assert x.shape[dim] == y.shape[dim], \
        f"X and Y must have same shape over dim {dim}."
    dim_size = x.shape[dim]

    def _sub_pairs(data: Tensor) -> Tensor:
        return data.swapaxes(-1, dim)\
            .unsqueeze(-1)\
            .repeat(*([1] * x.ndim), dim_size)\
            .sub(data.unsqueeze(-2))\
            .sign()

    res = _sub_pairs(x) * _sub_pairs(y) / (dim_size * (dim_size - 1))
    res = res.sum((-1, -2)).unsqueeze(-1).swapaxes(-1, dim)

    return res
