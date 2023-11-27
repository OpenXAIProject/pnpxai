import torch
from zennit.core import BasicHook, ParamMod, Stabilizer
from zennit.rules import NoMod

# TODO: better rule and composite for attention layer
class MultiheadAttentionRule(BasicHook):
    def __init__(self, stabilizer=1e-6):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod(param_keys=[])],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )

    # override backward: based on zennit.core.BasicHook.backward
    def backward(self, module, grad_input, grad_output):
        original_input = self.stored_tensors['input'][0].clone()
        inputs = []
        outputs = []
        for in_mod, param_mod, out_mod in zip(self.input_modifiers, self.param_modifiers, self.output_modifiers):
            input = in_mod(original_input).requires_grad_()
            with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
                output = modified.forward(input, input, input) # [MODIFIED] only if q == k == v
                output, _ = out_mod(output) # [MODIFIED] output
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
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)