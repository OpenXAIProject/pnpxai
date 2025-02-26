from torch import nn
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper, InputIdentity

from pnpxai.core.detector import symbolic_trace
from pnpxai.core.detector.utils import get_target_module_of, find_nearest_user_of
from pnpxai.core.detector.types import Convolution, Pool
from pnpxai.core.utils import ModelWrapper


def find_cam_target_layer(model: nn.Module) -> nn.Module:
    traced_model = symbolic_trace(model)
    last_conv_node = next(
        node for node in reversed(traced_model.graph.nodes)
        if isinstance(get_target_module_of(node), Convolution)
    )
    if last_conv_node is None:
        return
    pool_user = find_nearest_user_of(last_conv_node, Pool)
    if pool_user is None:
        return get_target_module_of(last_conv_node)
    conv_module_nm = next(iter(pool_user.prev.meta.get("nn_module_stack")))
    target_module = model
    for t in conv_module_nm.split("."):
        target_module = getattr(target_module, t)
    return target_module


class ModelWrapperForLayerAttribution(ModelInputWrapper):
    def __init__(
        self,
        wrapped_model: ModelWrapper,
    ):
        super().__init__(wrapped_model)

        # override
        self.arg_name_list = wrapped_model.required_order
        self.input_maps = nn.ModuleDict({
            arg_name: InputIdentity(arg_name) for arg_name in self.arg_name_list
        })
