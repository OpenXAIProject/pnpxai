from typing import Type, Callable, Sequence, Dict, Any, Optional, Union

from torch.nn import ModuleDict
from torch.nn.modules import Module
from zennit.canonizers import AttributeCanonizer

from transformers.models.visual_bert.modeling_visual_bert import VisualBertAttention

class ModuleConverter:
    """
    Converts an input module into multiple output modules based on specified configurations.

    Args:
        in_module (Module): The input module to be converted.
        out_modules (Dict[str, Dict]): A dictionary specifying the configurations for output modules.
        output_selector (Callable[[Dict], Any] | None, optional): A function to select output from converted modules.
    """

    def __init__(
        self,
        in_module: Module,
        out_modules: Dict[str, Dict],
        output_selector: Optional[Callable[[Dict], Any]]=None,
    ):
        self.in_module = in_module
        self.out_modules = out_modules
        self.output_selector = output_selector


    def convert_module(self):
        """
        Converts the input module into multiple output modules based on the specified configurations.

        Returns:
            ModuleDict: A dictionary containing the converted output modules.
        """
        module_dict = ModuleDict()
        for out_module_nm, cfg in self.out_modules.items():
            if cfg.get("module"):
                get_out_module = cfg["module"]
                out_module = get_out_module(self.in_module)
                module_dict.update({out_module_nm: out_module})
                continue
            # create out module instance
            out_module_type = cfg["out_module_type"]
            out_module_args = {
                out_module_arg_nm: get_out_module_arg(self.in_module)
                if isinstance(get_out_module_arg, Callable) else get_out_module_arg
                for out_module_arg_nm, get_out_module_arg in cfg["args"].items()
            }
            out_module = out_module_type(**out_module_args)

            # update output params
            in_params = self.in_module.state_dict()
            out_params = out_module.state_dict()
            for out_param_nm, get_output_param in cfg["params"].items():
                if isinstance(get_output_param, Callable):
                    out_params[out_param_nm] = get_output_param(in_params)
                elif isinstance(get_output_param, str):
                    input_param = in_params.get(get_output_param)
                    if input_param is not None:
                        out_params[out_param_nm] = input_param
            out_module.load_state_dict(out_params)
            module_dict.update({out_module_nm: out_module})
        if not self.in_module.training:
            module_dict.eval()
        return module_dict


    def convert_forward_args(
        self,
        out_module_name,
        in_args,
        in_kwargs,
        kept,
        out_module
    ):
        """
        Converts input arguments for the forward method of the specified output module.

        Args:
            out_module_name (str): The name of the output module.
            in_args: Input arguments for the forward method of the input module.
            in_kwargs: Input keyword arguments for the forward method of the input module.
            kept: The kept tensors or values.

        Returns:
            tuple: A tuple containing:
                - forward_args: The converted input arguments for the forward method of the specified output module.
                - forward_kwargs: The converted input keyword arguments for the forward method of the specified output module.
        """
        cfg = self.out_modules[out_module_name]["forward"]
        forward_args = cfg["args"](in_args, in_kwargs, kept, out_module)
        if cfg.get("kwargs"):
            forward_kwargs = {
                k: get_kwarg(in_args, in_kwargs, kept, out_module)
                if isinstance(get_kwarg, Callable) else get_kwarg
                for k, get_kwarg in cfg["kwargs"].items()
            }
        else:
            forward_kwargs = {}
        return forward_args, forward_kwargs
    
    def keep_args(self, out_module_name):
        """
        Determines whether to keep input arguments for the forward method of the specified output module.

        Args:
            out_module_name (str): The name of the output module.

        Returns:
            bool: True if input arguments should be kept, False otherwise.
        """
        return self.out_modules[out_module_name]["forward"].get("keep_args")

    def keep_kwargs(self, out_module_name):
        """
        Determines whether to keep input keyword arguments for the forward method of the specified output module.

        Args:
            out_module_name (str): The name of the output module.

        Returns:
            bool: True if input keyword arguments should be kept, False otherwise.
        """
        return self.out_modules[out_module_name]["forward"].get("keep_kwargs")
    
    def keep_outputs(self, out_module_name):
        """
        Determines whether to keep output tensors from the forward method of the specified output module.

        Args:
            out_module_name (str): The name of the output module.

        Returns:
            bool: True if output tensors should be kept, False otherwise.
        """
        return self.out_modules[out_module_name]["forward"].get("keep_outputs")

    def select_output(self, kept):
        """
        Selects the output from converted modules based on the specified output selector function.

        Args:
            kept: The kept tensors or values.

        Returns:
            Any: The selected output.
        """
        return self.output_selector(kept)
    

def _validate_and_warn(*args, cfg=None):
    if cfg.get("validations"):
        for validate in cfg["validations"]:
            validate(*args)
    if cfg.get("warnings"):
        for warn in cfg["warnings"]:

            warn(*args)


def _converted_forward(self, *in_args, **in_kwargs):
    """
    Forwards input to the converted modules.

    Args:
        *in_args: Input arguments for the forward method.
        **in_kwargs: Input keyword arguments for the forward method.

    Returns:
        Any: The output from the converted modules.
    """
    kept_args: Dict[str, Any] = {}
    kept_kwargs: Dict[str, Any] = {}
    kept_outputs: Dict[str, Any] = {}
    kept = {
        "args": kept_args,
        "kwargs": kept_kwargs,
        "outputs": kept_outputs,
    }
    for out_module_nm, out_module in self._converted_module.items():
        out_module_args, out_module_kwargs = self._converter.convert_forward_args(
            out_module_nm, in_args, in_kwargs, kept, self._converted_module,
        )
        out_module_outputs = out_module(*out_module_args, **out_module_kwargs)
        if self._converter.keep_args(out_module_nm):
            kept_args[out_module_nm] = out_module_args
        if self._converter.keep_kwargs(out_module_nm):
            kept_kwargs[out_module_nm] = out_module_kwargs
        if self._converter.keep_outputs(out_module_nm):
            kept_outputs[out_module_nm] = out_module_outputs
    
    if self._converter.output_selector is not None:
        outputs = self._converter.output_selector(kept)
        return outputs
    return out_module_outputs



# for typing
class ModuleConvertingCanonizer(AttributeCanonizer):
    """
    Canonizer for converting input module to multiple output modules.

    Attributes:
        in_module_type (Module): The input module type.
        config_selector (Callable): A function to select the configuration for output modules.
        out_module_configs (Dict): A dictionary specifying the configurations for output modules.
    """

    in_module_type: Module
    config_selector: Callable
    out_module_configs: Dict

    def __init__(self, attribute_map):
        super().__init__(attribute_map)
    

# factory
def module_converting_canonizer_factory(
        in_module_type: Type[Module],
        config_selector: Callable[[Module], Union[bool, Sequence[bool]]],
        out_module_configs: Dict[Union[bool, Sequence[bool]], Dict[str, Any]],
    ):
    """
    Factory function to create a ModuleConvertingCanonizer subclass.

    Args:
        in_module_type (Type[Module]): The input module type.
        config_selector (Callable[[Module], bool|Sequence[bool]]): A function to select the configuration for output modules.
        out_module_configs (Dict[bool|Sequence[bool], Dict[str, Any]]): A dictionary specifying the configurations for output modules.

    Returns:
        Type[ModuleConvertingCanonizer]: A subclass of ModuleConvertingCanonizer.
    """

    class ModuleConvertingCanonizerFromFactory(ModuleConvertingCanonizer):
        """
        Canonizer generated from the factory to convert input module to multiple output modules.
        Inherits from ModuleConvertingCanonizer.
        """
        def __init__(self):
            super().__init__(self._attribute_map)
        
        @classmethod
        def _attribute_map(cls, name, in_module):
            """
            Maps attributes for conversion.

            Args:
                name: The name of the input module.
                in_module: The input module to be converted.

            Returns:
                dict or None: Attributes for conversion.
            """
            if isinstance(in_module, cls.in_module_type):
                cfg_key = cls.config_selector(in_module)
                out_module_configs = cls.out_module_configs[cfg_key]
                converter = ModuleConverter(
                    in_module=in_module,
                    out_modules=out_module_configs["out_modules"],
                    output_selector=out_module_configs["output_selector"],
                )
                converted_module = converter.convert_module()
                attributes = {
                    "forward": cls.forward.__get__(in_module),
                    "_converter": converter,
                    "_converted_module": converted_module,
                }
                return attributes
            return None

        @staticmethod
        def forward(self, *in_args, **in_kwargs):
            return _converted_forward(self, *in_args, **in_kwargs)

    ModuleConvertingCanonizerFromFactory.in_module_type = in_module_type
    ModuleConvertingCanonizerFromFactory.config_selector = config_selector
    ModuleConvertingCanonizerFromFactory.out_module_configs = out_module_configs
    return ModuleConvertingCanonizerFromFactory
