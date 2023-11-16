import torch
from torch.utils.data import DataLoader

from zennit.core import BasicHook, Stabilizer, ParamMod
from zennit.composites import LayerMapComposite
from zennit.attribution import Gradient
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.types import Linear

from helpers import (
    get_torchvision_model,
    get_imagenet_dataset,
    denormalize_image,
    img_to_np,
)



class RapRule(BasicHook): # RelPropSimple
    def __init__(self, stabilizer=1e-6):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        input_modifiers = [
            lambda input: input.clamp(min=0), # px
            lambda input: input.clamp(min=0), # px
            lambda input: input.clamp(max=0), # nx
            lambda input: input.clamp(max=0), # nx
        ]
        param_modifiers = [
            ParamMod(modifier=lambda w, k: w.clamp(min=0), param_keys=["weight"]), # pw
            ParamMod(modifier=lambda w, k: w.clamp(max=0), param_keys=["weight"]), # nw
            ParamMod(modifier=lambda w, k: w.clamp(min=0), param_keys=["weight"]), # pw
            ParamMod(modifier=lambda w, k: w.clamp(max=0), param_keys=["weight"]), # nw
        ]
        # Then, outputs are = [pp(1), pn(2), np(3), nn(4)]

        def rap_gradient_mapper(out_grad, _outputs):
            global outputs
            print(out_grad)
            outputs = [out_grad.ne(0).type(out_grad.type()) * output for output in _outputs]
            _outputs = outputs[:2], outputs[2:]
            out_grad_p, out_grad_n = out_grad.clamp(min=0), out_grad.clamp(max=0)
            gradient_outputs = []
            for output in _outputs:
                s1p = out_grad_p * (1-1e-9) / stabilizer_fn(output[0])
                s1n = out_grad_n * (1-1e-9) / stabilizer_fn(output[0])
                gradient_outputs.append(s1p+s1n)
                s2p = out_grad_p * output[1] / stabilizer_fn(sum(output)) / stabilizer_fn(output[1])
                s2n = out_grad_n * output[1] / stabilizer_fn(sum(output)) / stabilizer_fn(output[1])
                gradient_outputs.append(s2p+s2n)
            return gradient_outputs
        
        def rap_reducer(inputs, gradients):
            print(torch.stack([input*gradient for input, gradient in zip(inputs, gradients)]).sum(1))
            return torch.stack([input*gradient for input, gradient in zip(inputs, gradients)]).sum(1)

        super().__init__(
            input_modifiers=input_modifiers,
            param_modifiers=param_modifiers,
            output_modifiers=[lambda output: output] * 4,
            gradient_mapper=(rap_gradient_mapper),
            reducer=(rap_reducer),
        )
        

model, transform = get_torchvision_model("resnet18")
dataset = get_imagenet_dataset(transform)
dataloader = DataLoader(dataset, batch_size=8)
inputs, labels = next(iter(dataloader))

canonizers = [SequentialMergeBatchNorm()]
composite = LayerMapComposite(canonizers=canonizers, layer_map=[(Linear, RapRule())])
with Gradient(model=model, composite=composite) as attributor:
    attributor(inputs, torch.eye(1000)[labels])
        # if torch.is_tensor(grad_output[0]) and grad_output[0].max() == 1:
        #     # first_prop (absolute influence prop)
        #     # C
        #     grad_outputs = self.gradient_mapper( # S = [S1, S2, S3, S4] = safe_divide(...)
        #         grad_output[0] * _outputs[0], # pd * Z
        #         _outputs, # Z
        #     )
        #     gradients = torch.autograd.grad( # C = [C1, C2, C3, C4] = gradprop(...)
        #         _outputs, # Z
        #         _inputs, # X
        #         grad_outputs=grad_outputs, # S
        #         create_graph=grad_output[0].requires_grad
        #     )

        #     # Cb
        #     safe_divide = lambda a, b: a / (b + b.eq(0) * 1e-9) * b.ne(0)
        #     p_ratio = safe_divide(sum(inputs[::2].sum(), inputs.sum()))
        #     n_ratio = safe_divide(sum(inputs[1::2].sum(), inputs.sum()))
        #     b = torch.stack([ # b = [bp, bn]
        #         module.bias * p_ratio,
        #         module.bias * n_ratio,
        #     ])
        #     _outputs_b = tuple([torch.stack(outputs[:2])]) # Zb = [Z1, Z2]
        #     _inputs_b = tuple(torch.stack(inputs[::2])) # Xb = [px, px]
        #     grad_outputs_b = self.gradient_mapper( # Sb = [Sb1, Sb2]
        #         grad_output[0] * b,
        #         _outputs_b,
        #     )
        #     gradients_b = torch.autograd.grad( # Cb = [Cb1, Cb2]
        #         _outputs_b, # Zb
        #         _inputs_b, # Xb
        #         grad_outputs = grad_outputs_b, # Sb
        #         create_graph=grad_output[0].requires_grad,
        #     )
        #     sb1 = self.gradient_mapper(bp, outputs[0])
        #     sb2 = self.gradient_mapper(bn, outputs[1])
        #     gradients += gradients_b

        #     # redistribute
        #     gp = gradients.clamp(min=0)
        #     gn = gradients.clamp(max=0)
        #     gdiff = (gp-gn).sum()
        #     gsum = (gp+gn).sum()
        #     gradients = (safe_divide(gp, gdiff) - safe_divide(gn, gdiff)) * gsum
        #     return gradients
        
        # safe_divide(gp, gtot) * 
        # gradients_tot = (gradients_p - gradients_n).sum()
        # relevance = self.reducer(inputs, gradients) # inputs * Cp
        # return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)
