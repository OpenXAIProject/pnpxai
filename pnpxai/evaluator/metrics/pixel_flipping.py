from typing import Callable, Optional, Union, Tuple, Dict, Any, Type, List

import torch
import torch.nn.functional as F
from torch.nn.modules import Module

from pnpxai.explainers.types import Tensor, TensorOrTupleOfTensors
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils.baselines import BaselineFunction, ZeroBaselineFunction
from pnpxai.utils import format_into_tuple, format_into_tuple_all
from pnpxai.evaluator.metrics.base import Metric


def default_prob_fn(outputs: torch.Tensor) -> torch.Tensor:
    return outputs.softmax(-1)


def default_pred_fn(outputs: torch.Tensor) -> torch.Tensor:
    return outputs.argmax(-1)


class PixelFlipping(Metric):
    """
    A metric class for evaluating the correctness of explanations or attributions provided by an explainer 
    using the pixel flipping technique.

    This class assesses the quality of attributions by perturbing input features (e.g., pixels) in the order 
    of their attributed importance and measuring the resulting change in the model's predictions. Correct attributions 
    should lead to significant changes in model predictions when the most important features are perturbed.

    Attributes:
        model (Module): The model.
        explainer (Optional[Explainer]=None): The explainer whose explanations are being evaluated.
        pooling_dim (int): Target channel dimension.
        n_steps (int): The number of perturbation steps.
        baseline_fn (Optional[BaselineFunction]): Function to generate baseline inputs for perturbation.
        prob_fn (Optional[Callable[[Tensor], Tensor]]): Function to compute probabilities from model outputs.
        pred_fn (Optional[Callable[[Tensor], Tensor]]): Function to compute predictions from model outputs.
        forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract forward arguments from inputs.
        additional_forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract additional forward arguments.
    
    Methods:
        evaluate(inputs, targets, attributions, attention_mask=None, descending=True):
            Evaluate the explainer's correctness based on the attributions by observing changes in model predictions.
    """

    def __init__(
        self,
        model: Module,
        explainer: Optional[Explainer] = None,
        pooling_dim: int = 1,
        n_steps: int = 10,
        baseline_fn: Optional[BaselineFunction] = None,
        prob_fn: Optional[Callable[[Tensor], Tensor]] = None,
        pred_fn: Optional[Callable[[Tensor], Tensor]] = None,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], torch.Tensor]] = None,
    ):
        super().__init__(
            model, explainer, target_input_keys,
            additional_input_keys, output_modifier
        )
        self.pooling_dim = pooling_dim
        self.n_steps = n_steps
        self.baseline_fn = baseline_fn or ZeroBaselineFunction()
        self.prob_fn = prob_fn or default_prob_fn
        self.pred_fn = pred_fn or default_pred_fn

    @torch.no_grad()
    def evaluate(
        self,
        inputs: TensorOrTupleOfTensors,
        targets: Tensor,
        attributions: TensorOrTupleOfTensors,
        attention_mask: Optional[TensorOrTupleOfTensors] = None,
        descending: bool = True,
    ) -> Union[Dict[str, Tensor], Tuple[Dict[str, Tensor]]]:
        """
        Evaluate the explainer's correctness based on the attributions by observing changes in model predictions.

        Args:
            inputs (TensorOrTupleOfTensors): The input tensors to the model.
            targets (Tensor): The target labels for the inputs.
            attributions (TensorOrTupleOfTensors): The attributions for the inputs.
            attention_mask (Optional[TensorOrTupleOfTensors], optional): Attention masks for the inputs. Default is None.
            descending (bool, optional): Whether to flip pixels in descending order of attribution. Default is True.

        Returns:
            Union[Dict[str, Tensor], Tuple[Dict[str, Tensor]]]: A dictionary or tuple of dictionaries containing
                the probabilities and predictions at each perturbation step.
        """
        forward_args, additional_forward_args = self.format_inputs(inputs)
        formatted: Dict[str, Tuple[Any]] = format_into_tuple_all(
            forward_args=forward_args,
            additional_forward_args=additional_forward_args,
            attributions=attributions,
            pooling_dim=self.pooling_dim or (None,)*len(format_into_tuple(forward_args)),
            baseline_fn=self.baseline_fn,
            attention_mask=attention_mask or (None,)*len(format_into_tuple(forward_args)),
        )
        assert all(
            len(formatted['forward_args']) == len(formatted[k]) for k in formatted
            if k != 'additional_forward_args'
        )

        bsz = formatted['forward_args'][0].size(0)
        results = []

        outputs = self._wrapped_model(
            *formatted['forward_args'],
            *formatted['additional_forward_args'],
        )
        init_probs = self.prob_fn(outputs)
        init_preds = self.pred_fn(outputs)

        for loc, forward_arg in enumerate(formatted['forward_args']):
            baseline_fn = formatted['baseline_fn'][loc]
            attrs = formatted['attributions'][loc]

            attrs, original_size = _flatten_if_not_1d(attrs)
            if formatted['attention_mask'][loc] is not None:
                attn_mask, _ = _flatten_if_not_1d(formatted['attention_mask'][loc])
                mask_value = -torch.inf if descending else torch.inf
                attrs = torch.where(attn_mask == 1, attrs, mask_value)

            valid_n_features = (~attrs.isinf()).sum(-1)
            n_flipped_per_step = valid_n_features // self.n_steps
            # ensure at least a pixel flipped
            n_flipped_per_step = n_flipped_per_step.clamp(min=1)
            sorted_indices = torch.argsort(
                attrs,
                descending=descending,
                stable=True,
            )
            probs = [_extract_target_probs(init_probs, targets)]
            preds = [init_preds]
            for step in range(1, self.n_steps):
                n_flipped = n_flipped_per_step * step
                if step + 1 == self.n_steps:
                    n_flipped = valid_n_features
                if any((n_flipped - 1) >= attrs.size(-1)):
                    # All features flipped already
                    # This break condition works when n_features < self.n_steps
                    break
                is_index_of_flipped = (
                    F.one_hot(n_flipped-1, num_classes=attrs.size(-1)).to(self.device)
                    .flip(-1).cumsum(-1).flip(-1)
                )
                is_flipped = _sort_by_order(
                    is_index_of_flipped, sorted_indices.argsort(-1))
                is_flipped = _recover_shape_if_flattened(is_flipped, original_size)
                is_flipped = _match_channel_dim_if_pooled(
                    is_flipped,
                    formatted['pooling_dim'][loc],
                    forward_arg.size()
                )

                baseline = baseline_fn(forward_arg)
                flipped_forward_arg = baseline * is_flipped + forward_arg * (1 - is_flipped)

                flipped_forward_args = tuple(
                    flipped_forward_arg if i == loc else formatted['forward_args'][i]
                    for i in range(len(formatted['forward_args']))
                )
                flipped_outputs = self._wrapped_model(
                    *flipped_forward_args,
                    *formatted['additional_forward_args'],
                )
                probs.append(_extract_target_probs(self.prob_fn(flipped_outputs), targets))
                preds.append(self.pred_fn(flipped_outputs))
            results.append({
                'probs': torch.stack(probs).transpose(1, 0),
                'preds': torch.stack(preds).transpose(1, 0),
            })
        if len(results) == 1:
            return results[0]
        return tuple(results)

    def _extract_forward_args(
        self,
        inputs: TensorOrTupleOfTensors,
    ) -> Tuple[Union[TensorOrTupleOfTensors, Type[None]]]:
        if isinstance(self.explainer, Explainer):
            return self.explainer._extract_forward_args(inputs)
        forward_args = self.forward_arg_extractor(inputs) \
            if self.forward_arg_extractor else inputs
        additional_forward_args = self.additional_forward_arg_extractor(inputs) \
            if self.additional_forward_arg_extractor else None
        return forward_args, additional_forward_args


class MoRF(PixelFlipping):
    """
    A metric class for evaluating the correctness of explanations or attributions using the 
    Most Relevant First (MoRF) pixel flipping technique.

    This class inherits from the PixelFlipping class and evaluates the quality of attributions by perturbing input 
    features (e.g., pixels) in descending order of their attributed importance. The average probability change is 
    measured to assess the explainer's correctness (lower better).

    Attributes:
        model (Module): The model.
        explainer (Optional[Explainer]=None): The explainer whose explanations are being evaluated.
        pooling_dim (int): Target channel dimension.
        n_steps (int): The number of perturbation steps.
        baseline_fn (Optional[BaselineFunction]): Function to generate baseline inputs for perturbation.
        prob_fn (Optional[Callable[[Tensor], Tensor]]): Function to compute probabilities from model outputs.
        pred_fn (Optional[Callable[[Tensor], Tensor]]): Function to compute predictions from model outputs.
        forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract forward arguments from inputs.
        additional_forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract additional forward arguments.
    
    Methods:
        evaluate(inputs, targets, attributions, attention_mask=None):
            Evaluate the explainer's correctness using the MoRF technique by observing changes in model predictions.
    """

    def __init__(
        self,
        model: Module,
        explainer: Optional[Explainer] = None,
        pooling_dim: int = 1,
        n_steps: int = 10,
        baseline_fn: Optional[BaselineFunction] = None,
        prob_fn: Optional[Callable[[Tensor], Tensor]] = None,
        pred_fn: Optional[Callable[[Tensor], Tensor]] = None,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], torch.Tensor]] = None,
    ):
        super().__init__(
            model, explainer, pooling_dim, n_steps,
            baseline_fn, prob_fn, pred_fn,
            target_input_keys,
            additional_input_keys,
            output_modifier,
        )

    def evaluate(
        self,
        inputs: TensorOrTupleOfTensors,
        targets: Tensor,
        attributions: TensorOrTupleOfTensors,
        attention_mask: Optional[TensorOrTupleOfTensors] = None
    ) -> TensorOrTupleOfTensors:
        """
        Evaluate the explainer's correctness using the MoRF technique by observing changes in model predictions.

        Args:
            inputs (TensorOrTupleOfTensors): The input tensors to the model.
            targets (Tensor): The target labels for the inputs.
            attributions (TensorOrTupleOfTensors): The attributions for the inputs.
            attention_mask (Optional[TensorOrTupleOfTensors], optional): Attention masks for the inputs. Default is None.

        Returns:
            TensorOrTupleOfTensors: The mean probabilities at each perturbation step, indicating the impact of 
                perturbing the most relevant features first.
        """
        pf_results = super().evaluate(inputs, targets, attributions, attention_mask, True)
        pf_results = format_into_tuple(pf_results)
        morf = tuple(result['probs'].mean(-1) for result in pf_results)
        if len(morf) == 1:
            morf = morf[0]
        return morf


class LeRF(PixelFlipping):
    """
    A metric class for evaluating the correctness of explanations or attributions using the 
    Least Relevant First (LeRF) pixel flipping technique.

    This class inherits from the PixelFlipping class and evaluates the quality of attributions by perturbing input 
    features (e.g., pixels) in ascending order of their attributed importance. The average probability change is 
    measured to assess the explainer's correctness.

    Attributes:
        model (Module): The model.
        explainer (Optional[Explainer]=None): The explainer whose explanations are being evaluated.
        pooling_dim (int): Target channel dimension.
        n_steps (int): The number of perturbation steps.
        baseline_fn (Optional[BaselineFunction]): Function to generate baseline inputs for perturbation.
        prob_fn (Optional[Callable[[Tensor], Tensor]]): Function to compute probabilities from model outputs.
        pred_fn (Optional[Callable[[Tensor], Tensor]]): Function to compute predictions from model outputs.
        forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract forward arguments from inputs.
        additional_forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract additional forward arguments.
    
    Methods:
        evaluate(inputs, targets, attributions, attention_mask=None):
            Evaluate the explainer's correctness using the LeRF technique by observing changes in model predictions.
    """

    def __init__(
        self,
        model: Module,
        explainer: Optional[Explainer] = None,
        pooling_dim: int = 1,
        n_steps: int = 10,
        baseline_fn: Optional[BaselineFunction] = None,
        prob_fn: Optional[Callable[[Tensor], Tensor]] = None,
        pred_fn: Optional[Callable[[Tensor], Tensor]] = None,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], torch.Tensor]] = None,
    ):
        super().__init__(
            model, explainer, pooling_dim, n_steps,
            baseline_fn, prob_fn, pred_fn,
            target_input_keys,
            additional_input_keys,
            output_modifier,
        )

    def evaluate(
        self,
        inputs: TensorOrTupleOfTensors,
        targets: Tensor,
        attributions: TensorOrTupleOfTensors,
        attention_mask: TensorOrTupleOfTensors = None,
    ) -> TensorOrTupleOfTensors:
        """
        Evaluate the explainer's correctness using the LeRF technique by observing changes in model predictions.

        Args:
            inputs (TensorOrTupleOfTensors): The input tensors to the model.
            targets (Tensor): The target labels for the inputs.
            attributions (TensorOrTupleOfTensors): The attributions for the inputs.
            attention_mask (Optional[TensorOrTupleOfTensors], optional): Attention masks for the inputs. Default is None.

        Returns:
            TensorOrTupleOfTensors: The mean probabilities at each perturbation step, indicating the impact of 
                perturbing the least relevant features first.
        """
        pf_results = super().evaluate(
            inputs, targets, attributions, attention_mask, False)
        pf_results = format_into_tuple(pf_results)
        lerf = tuple(result['probs'].mean(-1) for result in pf_results)
        if len(lerf) == 1:
            lerf = lerf[0]
        return lerf


class AbPC(PixelFlipping):
    """
    A metric class for evaluating the correctness of explanations or attributions using the 
    Area between Perturbation Curves (AbPC) technique.

    This class inherits from the PixelFlipping class and assesses the quality of attributions by comparing 
    the area between the perturbation curves obtained by perturbing input features (e.g., pixels) in both 
    ascending and descending order of their attributed importance. The average probability change is 
    measured, providing a comprehensive evaluation of the explainer's correctness.

    Attributes:
        model (Module): The model.
        explainer (Optional[Explainer]=None): The explainer whose explanations are being evaluated.
        pooling_dim (int): Target channel dimension.
        n_steps (int): The number of perturbation steps.
        baseline_fn (Optional[BaselineFunction]): Function to generate baseline inputs for perturbation.
        prob_fn (Optional[Callable[[Tensor], Tensor]]): Function to compute probabilities from model outputs.
        pred_fn (Optional[Callable[[Tensor], Tensor]]): Function to compute predictions from model outputs.
        forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract forward arguments from inputs.
        additional_forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract additional forward arguments.
        lb (float): The lower bound for clamping the probability differences.
    
    Methods:
        evaluate(inputs, targets, attributions, attention_mask=None):
            Evaluate the explainer's correctness using the AbPC technique by observing changes in model predictions.
    """

    def __init__(
        self,
        model: Module,
        explainer: Optional[Explainer]=None,
        pooling_dim: int=1,
        n_steps: int=10,
        baseline_fn: Optional[BaselineFunction]=None,
        prob_fn: Optional[Callable[[Tensor], Tensor]]=lambda outputs: outputs.softmax(-1),
        pred_fn: Optional[Callable[[Tensor], Tensor]]=lambda outputs: outputs.argmax(-1),
        lb: float=-1.,
        target_input_keys: Optional[List[Union[str, int]]]=None,
        additional_input_keys: Optional[List[Union[str, int]]]=None,
        output_modifier: Optional[Callable[[Any], torch.Tensor]]=None,
    ):
        super().__init__(
            model, explainer, pooling_dim, n_steps,
            baseline_fn, prob_fn, pred_fn,
            target_input_keys,
            additional_input_keys,
            output_modifier,
        )
        self.lb = lb

    def evaluate(
        self,
        inputs: TensorOrTupleOfTensors,
        targets: Tensor,
        attributions: TensorOrTupleOfTensors,
        attention_mask: Optional[TensorOrTupleOfTensors] = None,
        return_pf=False,
    ) -> TensorOrTupleOfTensors:
        """
        Evaluate the explainer's correctness using the AbPC technique by observing changes in model predictions.

        Args:
            inputs (TensorOrTupleOfTensors): The input tensors to the model.
            targets (Tensor): The target labels for the inputs.
            attributions (TensorOrTupleOfTensors): The attributions for the inputs.
            attention_mask (Optional[TensorOrTupleOfTensors], optional): Attention masks for the inputs.
            return_pf (Optional[bool]): Whether to return the perturbation curves for ascending and descending orders.

        Returns:
            TensorOrTupleOfTensors: The mean clamped differences in probabilities at each perturbation step, 
                indicating the impact of perturbing the most and least relevant features.
        """
        # pf by ascending order: lerf
        pf_ascs = super().evaluate(
            inputs, targets, attributions, attention_mask, False)
        pf_ascs = format_into_tuple(pf_ascs)

        # pf by descending order: morf
        pf_descs = super().evaluate(
            inputs, targets, attributions, attention_mask, True)
        pf_descs = format_into_tuple(pf_descs)

        # abpc
        results = []
        for pf_asc, pf_desc in zip(pf_ascs, pf_descs):
            result = (pf_asc['probs'] - pf_desc['probs']).clamp(min=self.lb).mean(-1)
            if return_pf:
                result = tuple([result, pf_desc, pf_asc])
            results.append(result)
        if len(results) == 1:
            return results[0]
        return tuple(results)


def _extract_target_probs(probs, targets):
    # please ensure probs.size() == (batch_size, n_classes)
    return probs[torch.arange(probs.size(0)), targets]


def _sort_by_order(x, permutation):
    d1, d2 = x.size()
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten().to(x.device),
        permutation.flatten().to(x.device)
    ].view(d1, d2)
    return ret


def _flatten_if_not_1d(batch):
    if batch.dim() > 2:
        original_size = batch.size()
        return batch.flatten(1), original_size
    return batch, batch.size()


def _recover_shape_if_flattened(batch, original_size):
    if batch.size() == original_size:
        return batch
    return batch.view(*original_size)


def _match_channel_dim_if_pooled(batch, pooling_dim, x_batch_size):
    if batch.size() == x_batch_size:
        return batch
    n_channels = x_batch_size[pooling_dim]
    n_repeats = tuple(
        n_channels if d == pooling_dim else 1
        for d in range(len(x_batch_size))
    )
    return batch.unsqueeze(pooling_dim).repeat(*n_repeats)
