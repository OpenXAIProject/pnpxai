#----------------------------
# .rules.py
#----------------------------

from zennit.rules import BasicHook, NoMod, ClampMod, zero_bias, ParamMod, Norm
from zennit.core import Stabilizer, expand


def absolute_influence_normalize(func):
    def wrapper(*args, **kwargs):
        relevance_raw = func(*args, **kwargs)[0]
        relsum = relevance_raw.sum(dim=-1, keepdim=True)
        absolute_influence = relevance_raw.abs() / relevance_raw.abs().sum(dim=-1, keepdim=True)
        relevance_normalized = relsum * absolute_influence
        print("ain", relevance_normalized.sum(dim=-1))
        return tuple([relevance_normalized])
    return wrapper


class AbsoluteInfluenceNormalization(BasicHook):
    def __init__(self, stabilizer=1e-6, zero_params=None):
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
            output_modifiers=[lambda output: output] * 4, # [pp, pn, np, nn]
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

        # z
        dim = tuple(range(1, original_input.dim()))
        relevance_ss_all = []
        for input, output in zip(inputs, outputs):
            gradients = torch.autograd.grad(
                (output,),
                (input,),
                grad_outputs=grad_output[0],
                create_graph=grad_output[0].requires_grad,
                retain_graph=True,
            )[0]
            relevance_ss = input * gradients
            relevance_ss_all.append(relevance_ss)
        relevance = sum(relevance_ss_all)

        # bias
        pos = ((outputs[0] + outputs[3]) * grad_output[0]).sum(dim=dim, keepdim=True)
        neg = ((outputs[1] + outputs[2]) * grad_output[0]).sum(dim=dim, keepdim=True)

        bp = module.bias * grad_output[0] * safe_divide(pos, pos+neg)
        bp_gradients = torch.autograd.grad(
            (outputs[0],),
            (inputs[0],),
            grad_outputs=safe_divide(bp, outputs[0]),
            create_graph=grad_output[0].requires_grad,
        )[0]
        bp_relevance = inputs[0] * bp_gradients
        relevance += bp_relevance

        bn = module.bias * grad_output[0] * safe_divide(neg, pos+neg)
        bn_gradients = torch.autograd.grad(
            (outputs[1],),
            (inputs[1],),
            grad_outputs=safe_divide(bn, outputs[1]),
            create_graph=grad_output[0].requires_grad,
        )[0]
        bn_relevance = inputs[0] * bn_gradients
        relevance += bn_relevance

        # redistribute
        relevance_sum = relevance.sum(dim=-1, keepdim=True)
        absolute_influence = relevance.abs() / relevance.abs().sum(dim=-1, keepdim=True)
        relevance_normalized = relevance_sum * absolute_influence
        print("ain", relevance.sum(dim=dim))
        return tuple(relevance_normalized if original.shape == relevance_normalized.shape else None for original in grad_input)
        
    # @absolute_influence_normalize    
    # def backward(self, module, grad_input, grad_output):
    #     return super().backward(module, grad_input, grad_output)




def safe_divide(a, b):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())

class RAPLinear(BasicHook):
    def __init__(self, stabilizer=1e-6, zero_params=None):
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
            output_modifiers=[lambda output: output.abs()] * 4, # [pp, pn, np, nn]
            gradient_mapper=(lambda out_grad, output: safe_divide(out_grad, output)),
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
        
        dim = tuple(range(1, original_input.dim()))
        relevance_s_all = []
        for i, (input, output) in enumerate(zip(inputs, outputs)):
            relevance_ss_stack = []
            output *= grad_output[0].ne(0)
            scale = 1 if i % 2 == 0 else (
                safe_divide(output, outputs[i-1]+output)
            )
            gradients_pos = torch.autograd.grad(
                (output,),
                (input,),
                grad_outputs=self.gradient_mapper(grad_output[0].clamp(min=0)*scale, output),
                create_graph=grad_output[0].requires_grad,
                retain_graph=True,
            )[0]
            gradients_neg = torch.autograd.grad(
                (output,),
                (input,),
                grad_outputs=self.gradient_mapper(grad_output[0].clamp(max=0)*scale, output),
                create_graph=grad_output[0].requires_grad,
            )[0]
            relevance_ss = self.reducer([input, input], [gradients_pos, gradients_neg])
            relevance_ss_stack.append(relevance_ss)
            if i % 2 == 1:
                relevance_s = sum(relevance_ss_stack)
                is_nonzero = relevance_s.ne(0)
                shift_numer = relevance_s.sum(dim=dim, keepdim=True) - grad_output[0].sum(dim=dim, keepdim=True)
                shift_denom = is_nonzero.sum(dim=dim, keepdim=True) * is_nonzero
                shift = safe_divide(shift_numer, shift_denom)
                relevance_s -= shift
                relevance_s_all.append(relevance_s)
                relevance_ss_stack = [] # clear
        relevance = sum(relevance_s_all)
        print("rap", relevance.sum(dim=dim))
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)


class RAPSum(BasicHook):
    def __init__(self, stabilizer=1e-6):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[lambda input: input.clamp(min=0)],
            param_modifiers=[NoMod(param_keys=[])],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )
    
    def backward(self, module, grad_input, grad_output):
        return super().backward(module, grad_input, grad_output)


class RAPBatchNorm2d(BasicHook):
    def __init__(self, stabilizer=1e-6, zero_params=None):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super.__init__(
            input_modifiers=[
                lambda input: input,
                torch.zeros_like, # bias
            ],
            param_modifiers=[
                NoMod(zero_params=zero_bias(zero_params=zero_params)),
                NoMod(zero_params=zero_params),
            ],
            output_modifers=[
                lambda output: output,
                lambda output: output,
            ],
            # gradient_mapper=(lambda out_grad, outputs: out_grad/stabilizer_fn(outputs[]))
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
                output = out_mod(output)
            inputs.append(input)
            outputs.append(output)
        # if module.__class__.__name__ == "Sum":
        #     import pdb; pdb.set_trace()
        gradients_pos = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=self.gradient_mapper(grad_output[0].clamp(min=0), outputs),
            create_graph=grad_output[0].requires_grad,
            retain_graph=True,
        )
        gradients_neg = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=self.gradient_mapper(grad_output[0].clamp(max=0), outputs),
            create_graph=grad_output[0].requires_grad
        )[0]
        # import pdb; pdb.set_trace()
        relevance = self.reducer(inputs, [gradients_pos[0]*gradients_neg])
        print(module.__class__.__name__, inputs[0].shape, relevance.shape)
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)



class ZBeta(BasicHook):
    def __init__(self, stabilizer=1e-6, zero_params=None):
        def sub(positive, *negatives):
            return positive - sum(negatives)

        mod_kwargs = {'zero_params': zero_params}
        stabilizer_fn = Stabilizer.ensure(stabilizer)

        super().__init__(
            input_modifiers=[
                lambda input: input,
                lambda input: expand(input.amin(dim=tuple(range(1,input.dim()))), input.shape, cut_batch_dim=True).to(input),
                lambda input: expand(input.amax(dim=tuple(range(1,input.dim()))), input.shape, cut_batch_dim=True).to(input),
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
        return ma.list_nodes()[-2].target
    
    def _find_last_prop_layer_name(self):
        ma = ModelArchitecture(self.model)
        return ma.list_nodes()[1].target

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
            (Linear, RAPLinear()),
            (Sum, RAPSum(stabilizer=1e-6)),
            (nn.BatchNorm2d, RAPBatchNorm2d(stabilizer=1e-6)),
        ] + layer_map_base(stabilizer=.25) + []
        name_map = [
            ((self._find_first_prop_layer_name(),), AbsoluteInfluenceNormalization()),
            ((self._find_last_prop_layer_name(),), ZBeta()),
        ]
        # import pdb; pdb.set_trace()
        canonizers = [SequentialMergeBatchNorm()]
        composite = NameLayerMapComposite(name_map=name_map, layer_map=layer_map, canonizers=canonizers)
        with Gradient(model=model, composite=composite) as attributor:
            _, relevance = attributor(inputs, torch.eye(n_classes)[targets].to(self.device))
        return relevance
