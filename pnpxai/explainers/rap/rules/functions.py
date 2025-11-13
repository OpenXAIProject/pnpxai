from pnpxai.explainers.rap.rules.base import (
    RelProp,
    RelPropSimple,
    _TensorOrTensors,
    safe_divide,
)
from torch import Tensor
import torch


class ReLU(RelProp):
    pass


class GeLU(RelProp):
    pass


class LeakyReLU(RelProp):
    pass


class SoftMax(RelProp):
    def relprop(
        self,
        rel: _TensorOrTensors,
        inputs: _TensorOrTensors,
        outputs: _TensorOrTensors,
        args=None,
        kwargs=None,
    ):
        dim = kwargs.get("dim", None)
        
        if dim is None:
            # if args[0] is tensor, then dim = args[1]
            if torch.is_tensor(args[0]) and len(args) > 1:
                dim = args[1]
            elif isinstance(args[0], int):
                dim = args[0]


        rel = (rel - (outputs * rel.sum(dim, keepdim=True))) * inputs
        return rel


class Add(RelPropSimple):
    def _partial_relprop(
        self,
        rel: Tensor,
        inputs: _TensorOrTensors,
        is_pos: bool,
        args=None,
        kwargs=None,
    ):
        clamp_kwargs = {"min": 0} if is_pos else {"max": 0}
        inputs = [torch.clamp(x, **clamp_kwargs) for x in inputs]
        pos_outputs = torch.add(*inputs)
        return super().relprop(rel, inputs, pos_outputs, args, kwargs)

    def relprop(
        self,
        rel: _TensorOrTensors,
        inputs: _TensorOrTensors,
        outputs: _TensorOrTensors,
        args=None,
        kwargs=None,
    ):
        if torch.is_tensor(inputs):
            return rel

        pos_rel = self._partial_relprop(rel, inputs, True, args, kwargs)
        neg_rel = self._partial_relprop(rel, inputs, False, args, kwargs)

        if torch.is_tensor(pos_rel):
            return pos_rel + neg_rel
        return [p + n for p, n in zip(pos_rel, neg_rel)]


class Sub(Add):
    pass


class Mul(RelPropSimple):
    def relprop(
        self,
        rel: _TensorOrTensors,
        inputs: _TensorOrTensors,
        outputs: _TensorOrTensors,
        args=None,
        kwargs=None,
    ):
        if torch.is_tensor(inputs):
            inputs = [inputs]

        inputs = [val for val in inputs if torch.is_tensor(val)]
        if len(inputs) <= 1:
            return rel

        return super().relprop(rel, inputs, outputs, args, kwargs)


class FloorDiv(Mul):
    pass


class Div(Mul):
    pass


class MatMul(RelProp):
    def relprop(
        self,
        rel: _TensorOrTensors,
        inputs: _TensorOrTensors,
        outputs: _TensorOrTensors,
        args=None,
        kwargs=None,
    ) -> _TensorOrTensors:
        in_a, in_b = inputs
        rel_norm = safe_divide(rel, outputs)

        rel_a = (rel_norm @ in_b.transpose(-1, -2)) * in_a
        rel_b = (in_a.transpose(-1, -2) @ rel_norm) * in_b

        return rel_a, rel_b


class Flatten(RelProp):
    def relprop(
        self,
        rel: _TensorOrTensors,
        inputs: _TensorOrTensors,
        outputs: _TensorOrTensors,
        args=None,
        kwargs=None,
    ):
        return rel.reshape(inputs.shape)


class Cat(RelPropSimple):
    def backward(
        self,
        rel: _TensorOrTensors,
        inputs: _TensorOrTensors,
        outputs: _TensorOrTensors,
        args=None,
        kwargs=None,
    ):
        Sp = safe_divide(rel, outputs)
        Cp = self.gradprop(outputs, inputs, Sp)
        rel = [x * cp for x, cp in zip(inputs, Cp)]

        return rel


class Repeat(RelPropSimple):
    pass


class GetItem(RelPropSimple):
    def relprop(
        self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None
    ):
        # If module's output is a tuple, propagate rel as is
        inputs = args[0]
        if not torch.is_tensor(inputs):
            return rel

        rel_fill = torch.zeros_like(inputs)
        rel_fill[args[1]] = rel

        return rel_fill


class Unsqueeze(RelProp):
    def relprop(
        self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None
    ) -> _TensorOrTensors:
        if "input" in kwargs:
            del kwargs["input"]
        else:
            args = args[1:]

        rel = torch.squeeze(rel, *args, **kwargs)
        return rel


class Squeeze(RelPropSimple):
    pass


class Expand(RelPropSimple):
    pass


class Permute(RelPropSimple):
    def relprop(
        self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None
    ) -> _TensorOrTensors:
        dims = kwargs.get("dims", None)
        if dims is None:
            dims = args[1] if isinstance(args[1], (tuple, list)) else args[1:]

        dims = torch.LongTensor(dims)
        inv = torch.empty_like(dims)
        inv[dims] = torch.arange(len(dims), device=dims.device)

        return rel.permute(inv.tolist())


class Reshape(RelProp):
    def relprop(
        self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None
    ) -> _TensorOrTensors:
        return rel.reshape(inputs.shape)


class Transpose(RelProp):
    def relprop(
        self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None
    ) -> Tensor:
        dim1 = kwargs.get("dim1", args[1] if len(args) > 1 else None)
        dim2 = kwargs.get("dim2", args[2] if len(args) > 2 else None)

        return rel.transpose(dim1, dim2)


class View(RelProp):
    def relprop(
        self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None
    ) -> Tensor:
        return rel.contiguous().view_as(inputs)


class GetAttr(RelProp):
    pass


class Contiguous(RelProp):
    pass


class Unfold(RelPropSimple):
    pass


class ReplicationPad1d(RelPropSimple):
    pass
