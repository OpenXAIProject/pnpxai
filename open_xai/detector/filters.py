def module_filter(n, module_name):
    return n.opcode == "call_module" and n.operator.__module__ == module_name

def conv_filter(n):
    return module_filter(n, "torch.nn.modules.conv")

def pool_filter(n):
    return module_filter(n, "torch.nn.modules.pooing")