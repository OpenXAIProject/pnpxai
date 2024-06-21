from torch import fx
from .types import ModuleType


def get_target_module_of(node: fx.Node):
    if node.op != "call_module":
        return
    target_module = node.graph._owning_module
    for i, t in enumerate(node.target.split(".")):
        if not hasattr(target_module, t):
            raise RuntimeError(
                f"Node referenced non-existent target {'.'.join(t[:i])}")
        target_module = getattr(target_module, t)
    return target_module


def find_nearest_user_of(node: fx.Node, module_type: ModuleType):
    for user in node.users:
        module = get_target_module_of(user)
        if isinstance(module, module_type):
            return user
    for user in node.users:
        uuser = find_nearest_user_of(user, module_type)
        if uuser is not None:
            return uuser
    return
