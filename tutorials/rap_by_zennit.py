from _operator import add
from abc import abstractmethod
from typing import Union, List, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from zennit.core import BasicHook, Stabilizer, ParamMod
from zennit.rules import NoMod

from zennit.composites import LayerMapComposite
from zennit.attribution import Gradient
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.types import Linear, Convolution
from zennit.layer import Sum

from pnpxai.detector import ModelArchitecture
from pnpxai.detector._core import NodeInfo
from pnpxai.explainers.lrp.utils import list_args_for_stack

from helpers import (
    get_torchvision_model,
    get_imagenet_dataset,
    denormalize_image,
    img_to_np,
)



output_modifiers = [
    lambda output, out_grad: output * out_grad.ne(0),
    lambda output, out_grad: output * out_grad.ne(0),
    lambda output, out_grad: output * out_grad.ne(0),
    lambda output, out_grad: output * out_grad.ne(0),
]


def gradient_mapper_builder(stabilizer_fn, first=False):
    if first:
        def first_gradient_mapper(out_grad, outputs, bias):
            outputs, outputs_b = outputs[:4], outputs[4:]
            _outputs = [output * out_grad for output in outputs]
            pos = (outputs[0] + outputs[3]).sum(dim=-1, keepdim=True)
            neg = (outputs[1] + outputs[2]).sum(dim=-1, keepdim=True)
            gradient_outputs = [
                _output / stabilizer_fn(output)
                for _output, output in zip(_outputs, outputs)
            ]
            gradient_outputs += [
                bias * out_grad * pos / stabilizer_fn(pos+neg),
                bias * out_grad * neg / stabilizer_fn(pos+neg)
            ]
            return gradient_outputs
        return first_gradient_mapper
    def gradient_mapper(out_grad, outputs):
        outputs = outputs[:2], outputs[2:]
        out_grad_p, out_grad_n = out_grad.clamp(min=0), out_grad.clamp(max=0)
        gradient_outputs = []
        for output in outputs:
            s1p = out_grad_p * (1-stabilizer_fn.epsilon) / stabilizer_fn(output[0])
            s1n = out_grad_n * (1-stabilizer_fn.epsilon) / stabilizer_fn(output[0])
            gradient_outputs.append(s1p+s1n)
            s2p = out_grad_p * output[1] / stabilizer_fn(sum(output)) / stabilizer_fn(output[1])
            s2n = out_grad_n * output[1] / stabilizer_fn(sum(output)) / stabilizer_fn(output[1])
            gradient_outputs.append(s2p+s2n)
        return gradient_outputs
    return gradient_mapper

def reducer_builder(stabilizer_fn, first=False, dim=-1):
    if first:
        def first_reducer(inputs, gradients):
            relevance = sum([input*gradient for input, gradient in zip(inputs, gradients)])

            # redist
            rel_p = relevance.clamp(min=0)
            rel_n = relevance.clamp(max=0)
            rel_diff = (rel_p - rel_n).sum(dim=dim, keepdim=True)
            rel_sum = (rel_p + rel_n).sum(dim=dim, keepdim=True)
            rel_redist = (
                rel_p / stabilizer_fn(rel_diff) * rel_sum
                - rel_n / stabilizer_fn(rel_diff) * rel_sum
            )
            return rel_redist
        return first_reducer
    def reducer(inputs, gradients, out_grad):
        # get relevance
        relevance = sum([input*gradient for input, gradient in zip(inputs, gradients)])

        # get shift
        diff = relevance.sum(dim=dim, keepdim=True) - out_grad.sum(dim=dim, keepdim=True)
        is_rel_nz = relevance.ne(0).type(relevance.type())
        shift = diff / stabilizer_fn(is_rel_nz.sum(dim=dim, keepdim=True)) * is_rel_nz
        return relevance - shift
    return reducer


class RapBaseRule(BasicHook):
    def __init__(
            self,
            stabilizer: float = 1e-9,
        ):
        self._stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers = self.input_modifiers,
            param_modifiers = self.param_modifiers,
            output_modifiers = self.output_modifiers,
            gradient_mapper = self.gradient_mapper,
            reducer = self.reducer,
        )
    
    @property
    def input_modifiers(self):
        return [
            lambda input: input.clamp(min=0), # px
            lambda input: input.clamp(min=0), # px
            lambda input: input.clamp(max=0), # nx
            lambda input: input.clamp(max=0), # nx
        ]

    @property
    def param_modifiers(self):
        return [
            ParamMod(modifier=lambda w, k: w.clamp(min=0), param_keys=["weight"]), # pw
            ParamMod(modifier=lambda w, k: w.clamp(max=0), param_keys=["weight"]), # nw
            ParamMod(modifier=lambda w, k: w.clamp(min=0), param_keys=["weight"]), # pw
            ParamMod(modifier=lambda w, k: w.clamp(max=0), param_keys=["weight"]), # nw
        ]
    
    @property
    @abstractmethod
    def output_modifiers(self):
        raise NotImplementedError

    @abstractmethod
    def gradient_mapper(self, *args):
        raise NotImplementedError

    @abstractmethod
    def reducer(self, *args):
        raise NotImplementedError
    
    @abstractmethod
    def _gmap_arg_selector(self, *args)
    def _get_gradient_mapper_args(self, *args)
        raise NotImplementedError

    def backward(self, module, grad_input, grad_output):
        '''Backward hook to compute LRP based on the class attributes.'''
        original_input = self.stored_tensors['input'][0].clone()
        inputs = []
        outputs = []
        for in_mod, param_mod, out_mod in zip(self.input_modifiers, self.param_modifiers, self.output_modifiers):
            input = in_mod(original_input).requires_grad_()
            with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
                output = modified.forward(input)
                output = out_mod(output)
            inputs.append(input)
            outputs.append(output)

        grad_outputs = self.gradient_mapper(grad_output[0], outputs)

        gradients = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            create_graph=grad_output[0].requires_grad
        )
        relevance = self.reducer(inputs, gradients)
        relevance = self._post_reducer(relevance)
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)


class RapFirstRule(RapBaseRule):
    def __init__(self, stabilizer: float=1e-9):
        super().__init__(
            stabilizer = stabilizer,
        )
    
    @property
    def output_modifiers(self):
        return [
            lambda output: output,
            lambda output: output,
            lambda output: output,
            lambda output: output,
        ]
    
    def gradient_mapper(self, out_grad, outputs, bias):
        outputs = outputs[:4]
        _outputs = [output * out_grad for output in outputs]
        pos = (outputs[0] + outputs[3]).sum(dim=-1, keepdim=True)
        neg = (outputs[1] + outputs[2]).sum(dim=-1, keepdim=True)
        gradient_outputs = [
            _output / self._stabilizer_fn(output)
            for _output, output in zip(_outputs, outputs)
        ]
        gradient_outputs += [
            bias * out_grad * pos / self._stabilizer_fn(pos+neg),
            bias * out_grad * neg / self._stabilizer_fn(pos+neg)
        ]
        return gradient_outputs
    
    def reducer(self, inputs, gradients):
        relevance = sum([input*gradient for input, gradient in zip(inputs, gradients)])
        # redist
        rel_p = relevance.clamp(min=0)
        rel_n = relevance.clamp(max=0)
        rel_diff = (rel_p - rel_n).sum(dim=dim, keepdim=True)
        rel_sum = (rel_p + rel_n).sum(dim=dim, keepdim=True)
        rel_redist = (
            rel_p / self._stabilizer_fn(rel_diff) * rel_sum
            - rel_n / self._stabilizer_fn(rel_diff) * rel_sum
        )
        return rel_redist

    def backward(self, module, grad_input, grad_output):
        original_input = self.stored_tensors['input'][0].clone()
        inputs = []
        outputs = []
        for in_mod, param_mod, out_mod in zip(self.input_modifiers, self.param_modifiers, self.output_modifiers):
            input = in_mod(original_input).requires_grad_()
            with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
                output = modified.forward(input)
                output = out_mod(output, grad_output[0])
            inputs.append(input)
            outputs.append(output)
        
        # create blanks for bias
        inputs += inputs[:2]
        outputs += outputs[:2]

        grad_outputs = self.gradient_mapper(grad_output[0], outputs, module.bias)
        gradients = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            create_graph=grad_output[0].requires_grad,
        )
        relevance = self.reducer(inputs, gradients, grad_output[0])
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)        


class RapRule(RapBaseRule): # RelPropSimple
    def __init__(self, stabilizer: float=1e-9):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            output_modifiers=output_modifiers,
            gradient_mapper=gradient_mapper_builder(stabilizer_fn, first=False),
            reducer=reducer_builder(stabilizer_fn, first=False),
        )

    def backward(self, module, grad_input, grad_output):
        '''Backward hook to compute LRP based on the class attributes.'''
        original_input = self.stored_tensors['input'][0].clone()
        inputs = []
        outputs = []
        for in_mod, param_mod, out_mod in zip(self.input_modifiers, self.param_modifiers, self.output_modifiers):
            input = in_mod(original_input).requires_grad_()
            with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
                output = modified.forward(input)
                output = out_mod(output, grad_output[0])
            inputs.append(input)
            outputs.append(output)
        
        grad_outputs = self.gradient_mapper(grad_output[0], outputs)
        gradients = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            create_graph=grad_output[0].requires_grad,
        )
        relevance = self.reducer(inputs, gradients, grad_output[0])
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)        


class RapLinearRule(RapRule):
    def __init__(self, stabilizer=1e-9):
        super().__init__(stabilizer=stabilizer, reducer_dim=-1)


class RapConvRule(RapRule):
    def __init__(self, stabilizer=1e-9):
        super().__init__(stabilizer=stabilizer, reducer_dim=[1,2,3])


model, transform = get_torchvision_model("resnet18")
dataset = get_imagenet_dataset(transform)
dataloader = DataLoader(dataset, batch_size=8)
inputs, labels = next(iter(dataloader))

canonizers = [SequentialMergeBatchNorm()]
composite = LayerMapComposite(canonizers=canonizers, layer_map=[
    (nn.Linear, RapLinearRule()),
    (nn.Conv2d, RapConvRule()),
])
with Gradient(model=model, composite=composite) as attributor:
    _, attrs = attributor(inputs, torch.eye(1000)[labels])
