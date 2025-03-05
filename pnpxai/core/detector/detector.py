from typing import Set, Tuple, Optional
from torch import fx, nn
from pnpxai.core._types import Model
from .utils import get_target_module_of
from .types import (
    ModuleType,
    Linear,
    Convolution,
    RNN,
    LSTM,
    Attention,
    Embedding,
)

DEFAULT_MODULE_TYPES_TO_DETECT = (
    Linear,
    Convolution,
    RNN,
    LSTM,
    Attention,
    Embedding,
)


class Tracer(fx.Tracer):
    # stop recursion when meeting timm's Attention
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        return (
            (
                m.__module__.startswith("torch.nn")
                or m.__module__.startswith("torch.ao.nn")
                or (
                    m.__module__ == "timm.models.vision_transformer"
                    and m.__class__.__name__ == "Attention"
                )
            ) and not isinstance(m, nn.Sequential)
        )


def symbolic_trace(model: nn.Module) -> fx.GraphModule:
    tracer = Tracer()
    graph = tracer.trace(model)
    name = (
        model.__class__.__name__ if isinstance(model, nn.Module) else model.__name__
    )
    return fx.GraphModule(tracer.root, graph, name)


def _get_node_target_module(node):
    target_module = node.graph.owning_module
    for t in node.target.splits('.'):
        target_module = getattr(target_module, t)
    return target_module


def _format_node_target(node) -> str:
    if node.op == 'get_attr':
        return f'getattr(self, {node.target})'
    elif node.op == 'call_function':
        return str(node.target)
    elif node.op == 'call_module':
        target_module = node.graph.owning_module
        for t in node.target.split('.'):
            target_module = getattr(target_module, t)
        return str(target_module)
    return node.target


def extract_graph_data(graph_module: fx.GraphModule):
    nodes, edges = [], []
    data = {'nodes': nodes, 'edges': edges}
    for node in graph_module.graph.nodes:
        nodes.append({
            'name': node.name,
            'op': node.op,
            'target': _format_node_target(node),
        })
        edges += [{
            'source': node.name,
            'target': user.name,
        } for user in node.users]
    return data    


def detect_model_architecture(
        model: Model,
        targets: Optional[Tuple[ModuleType]]=None,
    ) -> Set[ModuleType]:
    """
    A function detecting architecture for a given model.

    Args:
        model (Model): The machine learning model to be detected

    Returns:
        ModelArchitectureSummary: A summary of model architecture
    """
    targets = targets or DEFAULT_MODULE_TYPES_TO_DETECT
    detected = set()
    traced_model = symbolic_trace(model)
    for node in traced_model.graph.nodes:
        m = get_target_module_of(node)
        if m is None:
            continue
        tp = next((target for target in targets if isinstance(m, target)), None)
        if tp is None:
            continue
        detected.add(tp)
    return detected


def detect_model_architecture(
        model: Model,
        targets: Optional[Tuple[ModuleType]]=None,
    ) -> Set[ModuleType]:
    """
    A function detecting architecture for a given model.

    Args:
        model (Model): The machine learning model to be detected

    Returns:
        ModelArchitectureSummary: A summary of model architecture
    """
    targets = targets or DEFAULT_MODULE_TYPES_TO_DETECT
    detected = set()
    for nm, module in model.named_modules():
        module_type = next((target for target in targets if isinstance(module, target)), None)
        if module_type is None:
            continue
        detected.add(module_type)
    return detected