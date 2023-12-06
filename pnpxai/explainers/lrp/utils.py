# a function letting the model work on zennit, listing all args up
def list_args_for_stack(*args):
    assert all(arg.shape[1:] == args[0].shape[1:] for arg in args), "Cannot convert"
    if all(arg.shape == args[0].shape for arg in args):
        return [arg for arg in args]
    
    # followings are experimental: matching dimension of all args
    ls = []
    dim = args[0].dim()
    max_d = max(arg.shape[0] for arg in args)
    for arg in args:
        assert arg.shape[0] in [1, max_d], "Cannot convert"
        if arg.shape[0] == 1:
            arg = arg.repeat(max_d, *[1 for _ in range(dim-1)])
        ls.append(arg)
    return ls