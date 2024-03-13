#----------------------------
# .rules.py
#----------------------------

from zennit.rules import BasicHook, NoMod, ClampMod, zero_bias, ParamMod
from zennit.core import Stabilizer, expand

class AbsoluteInfluenceNormalization(BasicHook):
    def __init__(self, stabilizer=1e-6, zero_params=None):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0), # px
                lambda input: input.clamp(max=0), # nx
                lambda input: input.clamp(min=0), # px
                lambda input: input.clamp(max=0), # nx
            ],
            param_modifiers=[
                ClampMod(max=0, zero_params=zero_bias(zero_params)), # nw
                ClampMod(min=0, zero_params=zero_bias(zero_params)), # pw
                ClampMod(min=0, zero_params=zero_bias(zero_params)), # pw
                ClampMod(max=0, zero_params=zero_bias(zero_params)), # nw
            ],
            output_modifiers=[lambda output: output] * 6,
            gradient_mapper=(lambda out_grad, outputs: [out_grad/stabilizer_fn(output) for output in outputs]),
            reducer=(lambda inputs, gradients: sum([inp*grad for inp, grad in zip(inputs, gradients)])),
        )

    def backward(self, module, grad_input, grad_output):
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
            create_graph=grad_output[0].requires_grad,
            retain_graph=True,
        )
        relevance = self.reducer(inputs, gradients)

        # bias
        sum_neg = (sum(outputs[:2]) * grad_output[0]).sum(dim=-1, keepdim=True)
        sum_pos = (sum(outputs[2:]) * grad_output[0]).sum(dim=-1, keepdim=True)
        bias_neg = grad_output[0] * module.bias * sum_neg / (sum_neg + sum_pos + 1e-6)
        bias_pos = grad_output[0] * module.bias * sum_pos / (sum_neg + sum_pos + 1e-6)
        bias_grad_outputs = [bias_neg/outputs[2], bias_pos/outputs[0]]
        bias_gradients = torch.autograd.grad(
            [outputs[2], outputs[0]],
            [inputs[2], inputs[0]],
            grad_outputs=bias_grad_outputs,
            create_graph=grad_output[0].requires_grad,
        )
        relevance += self.reducer([inputs[2], inputs[0]], bias_gradients)

        # redistribute
        rel_pos = relevance.clamp(min=0)
        rel_neg = relevance.clamp(max=0)
        rel_tot = (rel_pos - rel_neg).sum(dim=-1, keepdim=True)

        rel_pos_redist = rel_pos / (rel_tot + 1e-6) * (rel_pos + rel_neg).sum(dim=-1, keepdim=True)
        rel_neg_redist = rel_neg / (rel_tot + 1e-6) * (rel_pos + rel_neg).sum(dim=-1, keepdim=True)
        relevance = rel_pos_redist + rel_neg_redist
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)


class RelativeAttributingPropagation(BasicHook):
    def __init__(self, stabilizer=1e-6, zero_params=None):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0), # px
                lambda input: input.clamp(min=0), # px
                lambda input: input.clamp(max=0), # nx
                lambda input: input.clamp(max=0), # nx
            ],
            param_modifiers=[
                ClampMod(min=0, zero_params=zero_bias(zero_params)), # pw
                ClampMod(max=0, zero_params=zero_bias(zero_params)), # nw
                ClampMod(min=0, zero_params=zero_bias(zero_params)), # pw
                ClampMod(max=0, zero_params=zero_bias(zero_params)), # nw
            ],
            output_modifiers=[lambda output: output.abs()] * 4,
            gradient_mapper=(lambda out_grad, output: out_grad / stabilizer_fn( output)),
            reducer=(lambda inputs, gradients: sum(input*gradient for input, gradient in zip(inputs, gradients))),
        )

    def backward(self, module, grad_input, grad_output):
        original_input = self.stored_tensors['input'][0].clone()
        sumdim = list(range(1, original_input.dim()))
        inputs = []
        outputs = []
        for in_mod, param_mod, out_mod in zip(self.input_modifiers, self.param_modifiers, self.output_modifiers):
            input = in_mod(original_input).requires_grad_()
            with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
                output = modified.forward(input)
                output = out_mod(output)
                output *= grad_output[0].ne(0)
            inputs.append(input)
            outputs.append(output)
        input_groups = inputs[:2], inputs[2:] # [[px, px], [nx, nx]]
        output_groups = outputs[:2], outputs[2:] # [[pp, pn], [np, nn]]

        out_grad_pos, out_grad_neg = grad_output[0].clamp(min=0), grad_output[0].clamp(max=0)
        _out_grads = [out_grad_pos, out_grad_neg, out_grad_pos, out_grad_neg]
        relevance = None
        for i, (inputs, outputs) in enumerate(zip(input_groups, output_groups)):
            gradient_groups = []
            _inputs = inputs + inputs
            output_pos, output_neg = outputs # [px, px, px, px]
            _outputs = [output_pos, output_pos, output_neg, output_neg] # [pp, pp, pn, pn]
            neg_scale = output_neg / (output_pos + output_neg + .25)
            _scales = [1, 1, neg_scale, neg_scale]
            _grad_outputs = [
                self.gradient_mapper(out_grad*scale, output)
                for out_grad, scale, output in zip(_out_grads, _scales, _outputs)
            ]
            gradient_groups.append(torch.autograd.grad(
                _outputs,
                _inputs,
                grad_outputs=_grad_outputs,
                create_graph=grad_output[0].requires_grad,
            ))
            _relevance = sum([self.reducer(inputs, gradients) for gradients in gradient_groups])
            shift_numerator = _relevance.sum(dim=sumdim, keepdim=True) - grad_output[0].sum(dim=sumdim, keepdim=True)
            is_nonzero_relevance = _relevance.ne(0)
            shift_denominator = is_nonzero_relevance.sum(dim=sumdim, keepdim=True) * is_nonzero_relevance
            shift = shift_numerator / (shift_denominator + .25)
            _relevance -= shift
            if relevance is None:
                relevance = _relevance
            else:
                relevance += _relevance
        # import pdb; pdb.set_trace()


        # grad_outputs = self.gradient_mapper(grad_output[0], outputs)
        # gradients = torch.autograd.grad(
        #     outputs,
        #     inputs,
        #     grad_outputs=grad_outputs,
        #     create_graph=grad_output[0].requires_grad,
        #     retain_graph=True,
        # )
        # relevance = self.reducer(inputs, gradients)
        print(relevance.sum())
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)



def _rap_gradient_mapper(out_grad, outputs):
    # inputs: p, p, n, n
    # outputs: pp, pn, np, nn
    # grad_outputs: rpos/pp, rneg/pp, neg_scale*rpos/pn, neg_scale*rneg/pn,
    #               rpos/np, rneg/np, neg_scale*rpos/nn, neg_scale*rneg/nn
    grad_outputs = []
    rel_pos = out_grad.clamp(min=0)
    rel_neg = out_grad.clamp(max=0)
    outputs = [output * out_grad.ne(0) for output in outputs]
    for i, output in enumerate(outputs):
        numerator = rel_pos if i % 2 == 0 else rel_neg
        if i % 4 > 1:
            numerator *= output / (outputs[i-2]+output+1e-6)
        grad_outputs.append(numerator/(output+1e-6))
    return grad_outputs


def _rap_reducer(inputs, gradients):
    # input x gradients
    relevance = sum(input*gradient for input, gradient in zip(inputs, gradients))

    # shift
    shift = relevance.sum(dim=-1, keepdim=True) / (relevance.ne(0).sum(axis=-1, keepdim=True) + 1e-6)
    relevance -= shift
    return relevance


class ZBeta(BasicHook):
    def __init__(self, stabilizer=1e-6, zero_params=None):
        def sub(positive, *negatives):
            return positive - sum(negatives)

        mod_kwargs = {'zero_params': zero_params}
        stabilizer_fn = Stabilizer.ensure(stabilizer)

        super().__init__(
            input_modifiers=[
                lambda input: input,
                lambda input: expand(input.min(dim=-1).values, input.shape, cut_batch_dim=True).to(input),
                lambda input: expand(input.max(dim=-1).values, input.shape, cut_batch_dim=True).to(input),
            ],
            param_modifiers=[
                NoMod(**mod_kwargs),
                ClampMod(min=0, **mod_kwargs),
                ClampMod(max=0, **mod_kwargs),
            ],
            output_modifiers=[lambda output: output] * 3,
            gradient_mapper=(lambda out_grad, outputs: (out_grad / stabilizer_fn(sub(*outputs)),) * 3),
            reducer=(lambda inputs, gradients: sub(*(input * gradient for input, gradient in zip(inputs, gradients)))),
        )


#----------------------------
# ./rap_zennit.py
#----------------------------

from _operator import add
from typing import List

import torch
from torch import nn
from captum._utils.typing import TargetType

from zennit.attribution import Gradient
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import LayerMapComposite, layer_map_base, NameLayerMapComposite
from zennit.rules import Epsilon, ZBox
from zennit.layer import Sum
from zennit.types import Linear

from pnpxai.core._types import Model, DataSource
from pnpxai.detector import ModelArchitecture
from pnpxai.detector._core import NodeInfo
from pnpxai.explainers._explainer import Explainer
from pnpxai.explainers.lrp.utils import list_args_for_stack

class RAPZennit(Explainer):
    def __init__(self, model: Model):
        super(RAPZennit, self).__init__(model)

    def _replace_add_func_with_mod(self):
        # get model architecture to manipulate
        ma = ModelArchitecture(self.model)

        # find add functions from model and replace them to modules
        add_func_nodes = ma.find_node(
            lambda n: n.operator is add and all(
                isinstance(arg, NodeInfo) for arg in n.args),
            get_all=True
        )
        if add_func_nodes:
            traced_model = self.__get_model_with_functional_nodes(ma, add_func_nodes)
            return traced_model
        return self.model

    def __get_model_with_functional_nodes(self, ma: ModelArchitecture, functional_nodes: List[NodeInfo]):
        for add_func_node in functional_nodes:
            add_mod_node = ma.replace_node(
                add_func_node,
                NodeInfo.from_module(Sum()),
            )
            stack_node = ma.insert_node(
                NodeInfo.from_function(torch.stack, dim=-1),
                add_mod_node,
                before=True,
            )
            _ = ma.insert_node(
                NodeInfo.from_function(list_args_for_stack),
                stack_node,
                before=True,
            )
        return ma.traced_model
    
    def _find_first_prop_layer_name(self):
        ma = ModelArchitecture(self.model)
        return ma.list_nodes()[-2].name
    
    def _find_last_prop_layer_name(self):
        ma = ModelArchitecture(self.model)
        return ma.list_nodes()[1].name

    def attribute(
        self,
        inputs: DataSource,
        targets: TargetType,
        epsilon: float = 1e-6,
        n_classes: int = None,
    ) -> List[torch.Tensor]:
        model = self._replace_add_func_with_mod()
        if isinstance(targets, int):
            targets = [targets] * len(inputs)
        elif torch.is_tensor(targets):
            targets = targets.tolist()
        else:
            raise Exception(f"[LRP] Unsupported target type: {type(targets)}")
        if n_classes is None:
            n_classes = self.model(inputs).shape[-1]
        
        layer_map = [
            (Linear, RelativeAttributingPropagation()),
        ] + layer_map_base()
        name_map = [
            ((self._find_first_prop_layer_name(),), AbsoluteInfluenceNormalization()),
            ((self._find_last_prop_layer_name(),), ZBeta()),
        ]
        canonizers = [SequentialMergeBatchNorm()]
        composite = NameLayerMapComposite(name_map=name_map, layer_map=layer_map, canonizers=canonizers)
        with Gradient(model=model, composite=composite) as attributor:
            _, relevance = attributor(inputs, torch.eye(n_classes)[targets].to(self.device))
        return relevance
