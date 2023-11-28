import torch
import torch.nn.functional as F
from zennit.core import BasicHook, ParamMod, Stabilizer, Hook
from zennit.rules import NoMod, Epsilon, Norm

# TODO: better rule and composite for attention layer
class AttentionHeadRule(BasicHook):
    def __init__(self, epsilon=1e-6, zero_params=None):
        stabilizer_fn = Stabilizer.ensure(epsilon)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod(zero_params=zero_params)],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )
    
    @staticmethod
    def conserving_forward(module, input):
        output, _ = module.forward(input, input, input)
        output.backward(input)
        return input * input.grad

    def backward(self, module, grad_input, grad_output):
        original_input = self.stored_tensors['input'][0].clone()
        inputs = []
        outputs = []
        _outputs = []

        for in_mod, param_mod, out_mod in zip(self.input_modifiers, self.param_modifiers, self.output_modifiers):
            input = in_mod(original_input).requires_grad_()
            with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
                output = self.conserving_forward(modified, input)
                # output, _ = module.forward(input, input, input)
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
        # import pdb; pdb.set_trace()
        print(module.__class__, grad_output[0].sum(), relevance.sum())
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)



class LayerNormRule(BasicHook):
    def __init__(self, stabilizer=1e-6):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod(param_keys=[])],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )
        self.stabilizer_fn = stabilizer_fn
    
    @staticmethod
    def conserving_forward(module, input):
        dims = list(range(1, input.dim()))
        mean = input.mean(dims)[(None,)*len(dims)].permute(*torch.arange(input.ndim-1, -1, -1))
        std = (input.var(dims) + 1e-6).sqrt()[(None,)*len(dims)].permute(*torch.arange(input.ndim-1, -1, -1))#.detach()
        normed = (input - mean) / std.detach()
        return normed #* module.weight + module.bias

    def backward(self, module, grad_input, grad_output):
        # grad_output[0] = R(y)
        original_input = self.stored_tensors['input'][0].clone()
        inputs = []
        outputs = []
        _outputs = []
        for in_mod, param_mod, out_mod in zip(self.input_modifiers, self.param_modifiers, self.output_modifiers):
            input = in_mod(original_input).requires_grad_()
            with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
                output = self.conserving_forward(modified, input) # normalize only: y = (x - mean(x)) / std(x): whatif weight * (x - mean(x)) / std(x) + bias case?
                # output = module.forward(input)
                output = out_mod(output)
            inputs.append(input)
            outputs.append(output)
        grad_outputs = self.gradient_mapper(grad_output[0], outputs) # R'(y) = R(y) / y

        # dR'(f(x))/dx = dy/dx * dR'(y)/dy
        gradients = torch.autograd.grad(
            outputs, # y
            inputs, # x
            grad_outputs=grad_outputs, # R
            create_graph=grad_output[0].requires_grad,
        )

        # R(x) = R'(f(x))/dx * x
        relevance = self.reducer(inputs, gradients)
        # import pdb; pdb.set_trace()
        print(module.__class__, grad_output[0].sum(), relevance.sum())
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)