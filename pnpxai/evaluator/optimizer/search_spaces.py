# from typing import Dict, Any, Optional, List

# import itertools
# from pnpxai.explainers.base import Explainer
# from pnpxai.explainers.utils.postprocess import RELEVANCE_POOLING_METHODS
# from pnpxai.explainers.utils.baselines import (
#     zero_baseline_function, token_baseline_function,
# )

# DEFAULT_SUGGEST_FUNCTION_CONFIGS_BASE = {
#     'n_iter': {
#         'method': 'suggest_int',
#         'low': 10,
#         'high': 100,
#         'step': 10,
#     },
#     'n_samples': {
#         'method': 'suggest_int',
#         'low': 10,
#         'high': 100,
#         'step': 10,
#     },
#     'n_steps': {
#         'method': 'suggest_int',
#         'low': 10,
#         'high': 100,
#         'step': 10,
#     },
#     'noise_level': {
#         'method': 'suggest_float',
#         'low': .1,
#         'high': 1.,
#         'step': .1,
#     },
#     'epsilon': {
#         'method': 'suggest_float',
#         'low': 1e-9,
#         'high': 1.,
#         'log': True,
#     },
#     'gamma': {
#         'method': 'suggest_float',
#         'low': 1e-9,
#         'high': 1.,
#         'log': True,
#     },
#     'pooling_method': {
#         'method': 'suggest_categorical',
#         'choices': list(RELEVANCE_POOLING_METHODS.keys()),
#     },
# }

# # baseline_fn
# def create_default_baseline_functions_for_image():
#     return {
#         'zeros': zero_baseline_function(),
#     }

# def create_default_baseline_functions_for_text(mask_token_id: int):
#     return {
#         'mask_token': token_baseline_function(token_id),
#     }

# def create_default_baseline_functions(
#     modality: ModalityOrTupleOfModalities,
#     mask_token_id: Optional[int]=None,
# ):
#     if modality == 'image':
#         return {
#             'zeros': zero_baseline_function(),
#             # add new baseline function here
#         }
#     elif modality == 'text':
#         return {
#             'mask_token': token_baseline_function(mask_token_id),
#             # add new baseline function here
#         }
#     elif isinstance(modality, tuple):
#         confs = [
#             create_default_baseline_functions(m, mask_token_id)
#             for m in modality
#         ]
#         combinated = {
#             '*'.join(comb): tuple(fns[k] for k, fns in zip(comb, confs))
#             for comb in itertools.product(*confs)
#         }
#         return combinated

# # postprocess
# def create_default_pooling_methods(
#     modality: ModalityOrTupleOfModalities
# ) -> List[str]:
#     if modality in ['image', 'text']:
#         return RELEVANCE_POOLING_METHODS
#     if isinstance(modality, tuple) and len(modality) > 1:
#         confs = [create_default_pooling_methods(m) for m in modality]
#         return ['*'.join(comb) for comb in itertools.product(*confs)]

# # fn_map
# def create_default_fn_map(
#     modality: ModalityOrTupleOfModalities,
#     mask_token_id: Optional[int]=None,
# ):
#     return {
#         'baseline_fn': create_default_baseline_functions(
#             modality, mask_token_id),
#     }

# # suggest methods
# def create_default_suggest_method_configs(
#     modality: ModalityOrTupleOfModalities,
#     mask_token_id: Optional[int]=None,
# ):
#     default_baseline_functions = create_default_baseline_functions(
#         modality, mask_token_id)
#     default_pooling_methods = create_default_pooling_methods(modality)
#     return {
#         **DEFAULT_SUGGEST_FUNCTION_CONFIGS_BASE,
#         'baseline_fn': {
#             'method': 'suggest_categorical',
#             'choices': list(default_baseline_functions.keys()),
#         },
#         'pooling_method': {
#             'method': 'suggest_categorical',
#             'choices': default_pooling_methods,
#         },
#     }


# DEFAULT_LOGSCALE_BASE = 10

# def _get_suggest_method(param_nm, func_conf):
#     def suggest(trial):
#         _func_conf = func_conf.copy()
#         method = _func_conf.pop('method')
#         return getattr(trial, method)(param_nm, **_func_conf)
#     return suggest

# def create_default_suggest_methods(
#     modality: ModalityOrTupleOfModalities,
#     explainer: Explainer,
#     mask_token_id: Optional[int]=None,
# ):
#     suggest_methods = {}
#     confs = create_default_suggest_method_configs(modality, mask_token_id)
#     for param_nm, func_conf in confs.items():
#         if hasattr(explainer, param_nm) or param_nm == 'pooling_method':
#             suggest_methods[param_nm] = _get_suggest_method(param_nm, func_conf)
#     return suggest_methods

# # search_space
# def create_default_search_space(
#     modality: ModalityOrTupleOfModalities,
#     explainer: Explainer,
# ):
#     search_space = {}
#     confs = create_default_suggest_method_configs(modality, mask_token_id)
#     for param_nm, func_conf in confs.items():
#         if hasattr(explainer, param_nm) or param_nm == 'pooling_method':
#             _func_conf = func_conf.copy()
#             method_nm = _func_conf.pop('method')
#             if method_nm == 'suggest_int':
#                 search_space[param_nm] = list(range(
#                     _func_conf['low'],
#                     _func_conf['high']+_func_conf['step'],
#                     _func_conf['step'],
#                 ))
#             elif method_nm == 'suggest_float':
#                 ls = [_func_conf['low']]
#                 while ls[-1] < _func_conf['high']:
#                     if _func_conf.get('log', False):
#                         ls.append(ls[-1]*DEFAULT_LOGSCALE_BASE)
#                     else:
#                         ls.append(ls[-1]+_func_conf['step'])
#                 search_space[param_nm] = ls
#             elif method_nm == 'suggest_categorical':
#                 search_space[param_nm] = _func_conf['choices']
#     return search_space
    
