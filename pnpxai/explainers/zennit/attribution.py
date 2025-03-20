from typing import Optional, Sequence, Union, List, Tuple, Callable, Any, Dict

from collections import defaultdict
import threading

import torch
import torchvision.transforms.functional as TF
from torch.nn import Module
from captum._utils.common import _run_forward, _sort_key_list, _reduce_list
from captum._utils.gradient import (
    compute_layer_gradients_and_eval,
    compute_gradients,
    apply_gradient_requirements,
    _extract_device_ids,
)
from zennit.attribution import Gradient as ZennitGradient
from zennit.core import Composite, Hook
from zennit.composites import LayerMapComposite
from zennit.types import Convolution, BatchNorm

from pnpxai.utils import format_into_tuple
from pnpxai.core._types import TensorOrTupleOfTensors, Tensor


class Gradient(ZennitGradient):
    def __init__(
        self,
        model: Module,
        composite: Optional[Composite] = None,
        attr_output=None,
        create_graph=False,
        retain_graph=None,
    ) -> None:
        super().__init__(
            model, composite, attr_output, create_graph, retain_graph)

    def grad(self, forward_args, targets, additional_forward_args=None):
        self._process_forward_args_before_grad(forward_args)
        grads = compute_gradients(
            self.model,
            forward_args,
            targets,
            additional_forward_args=additional_forward_args,
        )
        if len(grads) == 1:
            return grads[0]
        grads = tuple(g[0] for g in grads)
        return grads

    def forward(self, forward_args, targets, additional_forward_args=None):
        forward_args = self._process_forward_args_before_forward(forward_args)
        return self.grad(forward_args, targets, additional_forward_args)

    def _process_forward_args_before_forward(self, forward_args):
        if isinstance(forward_args, Sequence):
            return tuple(
                forward_arg.view_as(forward_arg) for forward_arg in forward_args
            )
        else:
            return forward_args.view_as(forward_args)

    def _process_forward_args_before_grad(self, forward_args):
        if isinstance(forward_args, Sequence):
            for forward_arg in forward_args:
                if forward_arg.is_floating_point() or forward_arg.is_complex():
                    forward_arg.requires_grad_()
        else:
            if forward_args.is_floating_point() or forward_args.is_complex():
                forward_args.requires_grad_()
        return forward_args


class LayerGradient(Gradient):
    def __init__(
        self,
        model: Module,
        layer: Union[str, Module, Sequence[Union[str, Module]]],
        composite: Optional[Composite] = None,
        attr_output=None,
        create_graph=False,
        retain_graph=None,
    ) -> None:
        super().__init__(
            model, composite, attr_output, create_graph, retain_graph)
        self.layer = layer

    def grad(self, forward_args, targets, additional_forward_args=None):
        grads, _ = compute_layer_gradients_and_eval(
            self.model,
            self.layer,
            forward_args,
            targets,
            additional_forward_args=additional_forward_args
        )
        if len(grads) == 1:
            return grads[0]
        grads = tuple(g[0] for g in grads)
        return grads


class SmoothGradient(Gradient):
    def __init__(
        self,
        model: Module,
        noise_level: Union[float, List[float]] = .1,
        n_iter: int = 20,
        composite: Optional[Composite] = None,
        attr_output=None,
        create_graph=None,
        retain_graph=None,
    ) -> None:
        super().__init__(
            model, composite, attr_output, create_graph, retain_graph)
        self.noise_level = noise_level
        self.n_iter = n_iter

    def forward(
        self,
        inputs: Tensor,
        targets: Tensor,
        additional_forward_args: Optional[TensorOrTupleOfTensors] = None,
        return_squared: bool = False,
    ):
        dims = tuple(range(1, inputs.ndim))
        std = self.noise_level * (inputs.amax(dims, keepdim=True) - inputs.amin(dims, keepdim=True))

        result = torch.zeros_like(inputs)
        result_sq = torch.zeros_like(inputs)
        for n in range(self.n_iter):
            if n == self.n_iter - 1:
                epsilon = torch.zeros_like(inputs)
            else:
                epsilon = torch.randn_like(inputs) * std
            grad = self.grad(
                inputs + epsilon,
                targets,
                additional_forward_args,
            )
            result += grad / self.n_iter
            if return_squared:
                result_sq += grad.pow(2) / self.n_iter
        if return_squared:
            return result, result_sq
        return result


class LayerSmoothGradient(LayerGradient):
    def __init__(
        self,
        model: Module,
        layer: Union[str, Module, List[Union[str, Module]]],
        noise_level: float=.1,
        n_iter: int=20,
        composite: Optional[Composite]=None,
        attr_output=None,
        create_graph=None,
        retain_graph=None,
    ) -> None:
        super().__init__(model, layer, composite, attr_output, create_graph, retain_graph)
        self.noise_level = noise_level
        self.n_iter = n_iter

    def grad(self, forward_args, targets, noise_fn, additional_forward_args=None):
        forward_args = self._process_forward_args_before_grad(forward_args)
        grads, _ = compute_layer_gradients_and_eval_with_noise(
            forward_fn=self.model,
            inputs=forward_args,
            layer=self.layer,
            noise_level=self.noise_level,
            noise_fn=noise_fn,
            target_ind=targets,
            additional_forward_args=additional_forward_args,
        )
        if len(grads) == 1:
            return grads
        grads = tuple(g[0] for g in grads)
        return grads

    def forward(
        self,
        forward_args: Union[Tensor, Tuple[Tensor]],
        targets: Tensor,
        additional_forward_args: Optional[Union[Tensor,Tuple[Tensor]]]=None,
        return_squared: bool=False,
    ) -> Union[Tensor, Tuple[Tensor]]:
        forward_args = self._process_forward_args_before_forward(forward_args)

        # forward to the layer
        forward_args = format_into_tuple(forward_args)
        for i in range(self.n_iter):
            noise_fn = torch.zeros_like if i == self.n_iter - 1 else torch.randn_like
            grads = self.grad(
                forward_args,
                targets,
                noise_fn,
                additional_forward_args=additional_forward_args,
            )
            if i == 0:
                results = tuple(grad/self.n_iter for grad in grads)
                if return_squared:
                    results_sq = tuple(grad.pow(2)/self.n_iter for grad in grads)
                continue
            results = tuple(
                result+grad/self.n_iter
                for result, grad in zip(results, grads)
            )
            if return_squared:
                results_sq = tuple(
                    result_sq+grad.pow(2)/self.n_iter
                    for result_sq, grad in zip(results, grads)
                )
        if return_squared:
            return results, results_sq
        return results


def _forward_layer_distributed_eval_with_noise(
    forward_fn: Callable,
    inputs: Any,
    layer: List[Union[str, Module]],
    noise_level: float,
    noise_fn: Callable=torch.randn_like,
    target_ind: Tensor = None,
    additional_forward_args: Any = None,
    attribute_to_layer_input: bool = False,
    forward_hook_with_return: bool = True,
    require_layer_grads: bool = True,
) -> Union[
    Tuple[Dict[Module, Dict[torch.device, Tuple[Tensor, ...]]], Tensor],
    Dict[Module, Dict[torch.device, Tuple[Tensor, ...]]],
]:
    r"""
    A helper function that allows to set a hook on model's `layer`, run the forward
    pass and returns intermediate layer results, stored in a dictionary,
    and optionally also the output of the forward function. The keys in the
    dictionary are the device ids and the values are corresponding intermediate layer
    results, either the inputs or the outputs of the layer depending on whether we set
    `attribute_to_layer_input` to True or False.
    This is especially useful when we execute forward pass in a distributed setting,
    using `DataParallel`s for example.
    """
    saved_layer: Dict[Module, Dict[device, Tuple[Tensor, ...]]] = defaultdict(dict)
    lock = threading.Lock()
    all_layers: List[Module] = [layer] if isinstance(layer, Module) else layer

    # Set a forward hook on specified module and run forward pass to
    # get layer output tensor(s).
    # For DataParallel models, each partition adds entry to dictionary
    # with key as device and value as corresponding Tensor.
    def hook_wrapper(original_module):
        def forward_hook(module, inp, out=None):
            eval_tsrs = inp if attribute_to_layer_input else out
            is_eval_tuple = isinstance(eval_tsrs, tuple)

            if not is_eval_tuple:
                eval_tsrs = (eval_tsrs,)

            # add noise
            dims = tuple(tuple(range(1, eval_tsr.ndim)) for eval_tsr in eval_tsrs)
            stds = tuple(
                noise_level * (eval_tsr.amax(dim=dim, keepdim=True) - eval_tsr.amin(dim=dim, keepdim=True))
                for eval_tsr, dim in zip(eval_tsrs, dims)
            )
            eval_tsrs = tuple(
                eval_tsr + noise_fn(eval_tsr) * std
                for eval_tsr, std in zip(eval_tsrs, stds)
            )

            if require_layer_grads:
                apply_gradient_requirements(eval_tsrs, warn=False)
            with lock:
                nonlocal saved_layer
                # Note that cloning behaviour of `eval_tsr` is different
                # when `forward_hook_with_return` is set to True. This is because
                # otherwise `backward()` on the last output layer won't execute.
                if forward_hook_with_return:
                    saved_layer[original_module][eval_tsrs[0].device] = eval_tsrs
                    eval_tsrs_to_return = tuple(
                        eval_tsr.clone() for eval_tsr in eval_tsrs
                    )
                    if not is_eval_tuple:
                        eval_tsrs_to_return = eval_tsrs_to_return[0]
                    return eval_tsrs_to_return
                else:
                    saved_layer[original_module][eval_tsrs[0].device] = tuple(
                        eval_tsr.clone() for eval_tsr in eval_tsrs
                    )

        return forward_hook

    all_hooks = []
    try:
        for single_layer in all_layers:
            if attribute_to_layer_input:
                all_hooks.append(
                    single_layer.register_forward_pre_hook(hook_wrapper(single_layer))
                )
            else:
                all_hooks.append(
                    single_layer.register_forward_hook(hook_wrapper(single_layer))
                )
        output = _run_forward(
            forward_fn,
            inputs,
            target=target_ind,
            additional_forward_args=additional_forward_args,
        )
    finally:
        for hook in all_hooks:
            hook.remove()

    if len(saved_layer) == 0:
        raise AssertionError("Forward hook did not obtain any outputs for given layer")

    if forward_hook_with_return:
        return saved_layer, output
    return saved_layer


def compute_layer_gradients_and_eval_with_noise(
    forward_fn: Callable,
    layer: List[Union[str, Module]],
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    noise_level: float,
    noise_fn: Callable=torch.randn_like,
    target_ind: Optional[Tensor] = None,
    additional_forward_args: Any = None,
    gradient_neuron_selector: Union[
        None, int, Tuple[Union[int, slice], ...], Callable
    ] = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
    output_fn: Union[None, Callable] = None,
    grad_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[List[Tuple[Tensor, ...]], List[Tuple[Tensor, ...]]],
]:
    r"""
    Computes gradients of the output with respect to a given layer as well
    as the output evaluation of the layer for an arbitrary forward function
    and given input.

    For data parallel models, hooks are executed once per device ,so we
    need to internally combine the separated tensors from devices by
    concatenating based on device_ids. Any necessary gradients must be taken
    with respect to each independent batched tensor, so the gradients are
    computed and combined appropriately.

    More information regarding the behavior of forward hooks with DataParallel
    models can be found in the PyTorch data parallel documentation. We maintain
    the separate inputs in a dictionary protected by a lock, analogous to the
    gather implementation for the core PyTorch DataParallel implementation.

    NOTE: To properly handle inplace operations, a clone of the layer output
    is stored. This structure inhibits execution of a backward hook on the last
    module for the layer output when computing the gradient with respect to
    the input, since we store an intermediate clone, as
    opposed to the true module output. If backward module hooks are necessary
    for the final module when computing input gradients, utilize
    _forward_layer_eval_with_neuron_grads instead.

    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        layer:      Layer for which gradients / output will be evaluated.
        inputs:     Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        output_fn:  An optional function that is applied to the layer inputs or
                    outputs depending whether the `attribute_to_layer_input` is
                    set to `True` or `False`
        args:       Additional input arguments that forward function requires.
                    It takes an empty tuple (no additional arguments) if no
                    additional arguments are required
        grad_kwargs: Additional keyword arguments for torch.autograd.grad


    Returns:
        tuple[**gradients**, **evals**]:
        - **gradients**:
            Gradients of output with respect to target layer output.
        - **evals**:
            Target layer output for given input.
    """
    with torch.autograd.set_grad_enabled(True):
        # saved_layer is a dictionary mapping device to a tuple of
        # layer evaluations on that device.
        saved_layer, output = _forward_layer_distributed_eval_with_noise(
            forward_fn,
            inputs,
            layer,
            noise_level,
            noise_fn,
            target_ind=target_ind,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
            forward_hook_with_return=True,
            require_layer_grads=True,
        )
        assert output[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )

        device_ids = _extract_device_ids(forward_fn, saved_layer, device_ids)

        # Identifies correct device ordering based on device ids.
        # key_list is a list of devices in appropriate ordering for concatenation.
        # If only one key exists (standard model), key list simply has one element.
        key_list = _sort_key_list(
            list(next(iter(saved_layer.values())).keys()), device_ids
        )
        all_outputs: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]
        if isinstance(layer, Module):
            all_outputs = _reduce_list(
                [
                    (
                        saved_layer[layer][device_id]
                        if output_fn is None
                        else output_fn(saved_layer[layer][device_id])
                    )
                    for device_id in key_list
                ]
            )
        else:
            all_outputs = [
                _reduce_list(
                    [
                        (
                            saved_layer[single_layer][device_id]
                            if output_fn is None
                            else output_fn(saved_layer[single_layer][device_id])
                        )
                        for device_id in key_list
                    ]
                )
                for single_layer in layer
            ]
        all_layers: List[Module] = [layer] if isinstance(layer, Module) else layer
        grad_inputs = tuple(
            layer_tensor
            for single_layer in all_layers
            for device_id in key_list
            for layer_tensor in saved_layer[single_layer][device_id]
        )
        saved_grads = torch.autograd.grad(
            outputs=torch.unbind(output),
            inputs=grad_inputs,
            **grad_kwargs or {},
        )

        offset = 0
        all_grads: List[Tuple[Tensor, ...]] = []
        for single_layer in all_layers:
            num_tensors = len(next(iter(saved_layer[single_layer].values())))
            curr_saved_grads = [
                saved_grads[i : i + num_tensors]
                for i in range(
                    offset, offset + len(key_list) * num_tensors, num_tensors
                )
            ]
            offset += len(key_list) * num_tensors
            if output_fn is not None:
                curr_saved_grads = [
                    output_fn(curr_saved_grad) for curr_saved_grad in curr_saved_grads
                ]

            all_grads.append(_reduce_list(curr_saved_grads))

        layer_grads: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]
        layer_grads = all_grads
        if isinstance(layer, Module):
            layer_grads = all_grads[0]

        if gradient_neuron_selector is not None:
            assert isinstance(
                layer, Module
            ), "Cannot compute neuron gradients for multiple layers simultaneously!"
            inp_grads = _neuron_gradients(
                inputs, saved_layer[layer], key_list, gradient_neuron_selector
            )
            return (
                cast(Tuple[Tensor, ...], layer_grads),
                cast(Tuple[Tensor, ...], all_outputs),
                inp_grads,
            )
    return layer_grads, all_outputs  # type: ignore


def _get_bias_data(module):
    # Borrowed from official paper impl:
    # https://github.com/idiap/fullgrad-saliency/blob/master/saliency/tensor_extractor.py#L47
    if isinstance(module, BatchNorm):
        bias = -(module.running_mean*module.weight
            /(module.running_var+module.eps).sqrt())+module.bias
        return bias.data
    elif module.bias is not None:
        return module.bias.data
    return None


class _SavingBias(Hook):
    def forward(self, module, input, output):
        self.stored_tensors['bias'] = _get_bias_data(module)


class FullGradient(ZennitGradient):
    def __init__(
        self,
        model: Module,
        pooling_method: Optional[Callable[[Tensor], Tensor]]=None,
        interpolate_mode: Optional[TF.InterpolationMode]=None,
    ):
        layer_map = [(tp, _SavingBias()) for tp in [Convolution, BatchNorm]]
        composite = LayerMapComposite(layer_map=layer_map)
        super().__init__(model, composite, None)
        self.pooling_method = pooling_method
        self.interpolate_mode = interpolate_mode

    def forward(self, input, attr_output_fn):
        input = input.view_as(input)
        _, gradient = self.grad(input, attr_output_fn)
        rels = [gradient * input] # gradient x input
        for hook in self.composite.hook_refs:
            if hook.stored_tensors['bias'] is not None:
                grad_x_bias = (
                    hook.stored_tensors['bias'][None, :, None, None]
                    * hook.stored_tensors['grad_output'][0]
                )
                grad_x_bias = TF.resize(
                    grad_x_bias,
                    input.shape[2:],
                    interpolation=self.interpolate_mode,
                )
                rels.append(grad_x_bias)
        pooled = [self.pooling_method(rel) for rel in rels]
        return sum(pooled)[:, None, :, :]
