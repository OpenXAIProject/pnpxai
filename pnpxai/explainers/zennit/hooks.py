from zennit.core import RemovableHandleList, RemovableHandle, Hook, BasicHook

class HookWithKwargs(Hook):
    '''Base class for hooks to be used to compute layer-wise attributions.'''
    def __init__(self):
        super().__init__()
        self.stored_kwargs = None
    
    def pre_forward(self, module, input, kwargs):
        return super().pre_forward(module, input), kwargs

    def forward(self, module, input, kwargs, output):
        '''Forward hook to save module in-/outputs.'''
        self.stored_tensors['input'] = input
        self.stored_kwargs = kwargs

    def register(self, module):
        '''Register this instance by registering all hooks to the supplied module.'''
        return RemovableHandleList([
            RemovableHandle(self),
            module.register_forward_pre_hook(self.pre_forward, with_kwargs=True),
            module.register_forward_hook(self.post_forward),
            module.register_forward_hook(self.forward, with_kwargs=True),
        ])


# myeongjin hi
class BasicHookWithRelevanceModifier(BasicHook):
    def __init__(
    ) -> None:
        pass

