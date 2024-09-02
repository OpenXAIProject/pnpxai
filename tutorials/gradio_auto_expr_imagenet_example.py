import torch

#------------------------------------------------------------------------------#
#----------------------------------- setup ------------------------------------#
#------------------------------------------------------------------------------#

from helpers import load_model_and_dataloader_for_tutorial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, loader, _ = load_model_and_dataloader_for_tutorial('image', device)


#------------------------------------------------------------------------------#
#---------------------------- auto experiment ---------------------------------#
#------------------------------------------------------------------------------#

from pnpxai import AutoExplanationForImageClassification

expr = AutoExplanationForImageClassification(
    model=model,
    data=loader,
    input_extractor=lambda batch: batch[0].to(device),
    label_extractor=lambda batch: batch[1].to(device),
    target_extractor=lambda outputs: outputs.argmax(-1).to(device),
    channel_dim=1,
)


# run_batch returns a dict of results
results = expr.run_batch(data_ids=[0, 1], explainer_id=1, postprocessor_id=0, metric_id=1)

# Also, we can use experiment manager to browse results as follows
# get data
data = expr.manager.get_data_by_id(0) # a pair of single instance (input, label)
batch = expr.manager.batch_data_by_ids(data_ids=[0, 1]) # a batch of multiple instances (inputs, labels)

# get explainer
explainer = expr.manager.get_explainer_by_id(0)

# get explanation
attr = expr.manager.get_explanation_by_id(data_id=0, explainer_id=1)
batched_attrs = expr.manager.batch_explanations_by_ids(data_ids=[0,1], explainer_id=1) # batched explanations

# get postprocessor
postprocessor = expr.manager.get_postprocessor_by_id(0)

# postprocess
postprocessed = postprocessor(batched_attrs) # Note that this work only for batched attrs

# get metric
metric = expr.manager.get_metric_by_id(0)

# get evaluation
evaluation = expr.manager.get_evaluation_by_id(
    data_id=0,
    explainer_id=1,
    postprocessor_id=0,
    metric_id=1,
)
evaluations = expr.manager.batch_evaluations_by_ids( # batched evaluations
    data_ids=[0, 1],
    explainer_id=1,
    postprocessor_id=0,
    metric_id=1,
)



# #------------------------------------------------------------------------------#
# #-------------------------------------- app -----------------------------------#
# #------------------------------------------------------------------------------#

# import gradio as gr

# # clear results for manual experiment by UI
# expr.manager.clear()

# # maps
# nm2cls_explainer = {
#     cls.__name__: cls for cls in expr.recommended.explainers
# }
# nm2obj_postprocessor = {
#     pp.pooling_method: pp for pp in expr.manager.postprocessors
# }
# nm2obj_baseline_fn = {
#     nm: None for nm in ['zeros', 'min', 'channel_min']
# }

# # forms
# def change_form_explainer_kwargs(explainer_nm):
#     visible = {
#         'n_steps': explainer_nm in ['IntegratedGradients'],
#         'n_iter': explainer_nm in ['SmoothGrad', 'VarGrad'],
#         'n_samples': explainer_nm in ['Lime', 'KernelShap'],
#         'baseline_fn': explainer_nm in ['IntegratedGradients', 'Lime', 'KernelShap'],
#         'noise_level': explainer_nm in ['SmoothGrad', 'VarGrad'],
#         'epsilon': 'LRP' in explainer_nm,
#         'gamma': explainer_nm in ['LRPEpsilonGammaBox'],
#         'pooling_method': True,
#     }
#     form = {
#         'n_steps': gr.Number(
#             25,
#             label='n_steps',
#             info='The number of approximation steps',
#             visible=visible['n_steps'],
#             interactive=True,
#         ),
#         'n_iter': gr.Number(
#             25,
#             label='n_iter',
#             info='The number of iterations',
#             visible=visible['n_iter'],
#             interactive=True,
#         ),
#         'n_samples': gr.Number(
#             25,
#             label='n_samples',
#             info='The number of samples',
#             visible=visible['n_samples'],
#             interactive=True,
#         ),
#         'baseline_fn': gr.Dropdown(
#             list(nm2obj_baseline_fn),
#             label='baseline_fn',
#             info='Baseline function',
#             visible=visible['baseline_fn'],
#             interactive=True,
#         ),
#         'noise_level': gr.Slider(
#             minimum=0., maximum=1., value=.1, step=.05,
#             label='noise_level',
#             info='Noise level',
#             visible=visible['noise_level'],
#             interactive=True,
#         ),
#         'epsilon': gr.Slider(
#             minimum=0., maximum=1., value=.1, step=.05,
#             label='epsilon',
#             info='Epsilon',
#             visible=visible['epsilon'],
#             interactive=True,
#         ),
#         'gamma': gr.Slider(
#             minimum=0., maximum=1., value=.25, step=.05,
#             label='gamma',
#             info='Gamma',
#             visible=visible['gamma'],
#             interactive=True,
#         ),
#         'pooling_method': gr.Dropdown(
#             list(nm2obj_postprocessor.keys()),
#             label='pooling_method',
#             info='Relevance Pooling Method',
#             visible=visible['pooling_method'],
#             interactive=True,
#         ),
#     }
#     return list(form.values())


# with gr.Blocks() as demo:
#     explainers = gr.State([])
#     gr.Markdown('## Explainers')
#     with gr.Row():
#         with gr.Column():
#             current_kwargs = gr.State([])

#             # create a form of explainer type
#             new_explainer_nm = gr.Dropdown(
#                 list(nm2cls_explainer.keys()),
#                 label='Explainer Type',
#                 info='Recommended Explainers'
#             )

#             # create forms of explainer kwargs by explainer type
#             form_explainer_kwargs = change_form_explainer_kwargs('')
#             new_explainer_nm.change(
#                 change_form_explainer_kwargs,
#                 new_explainer_nm,
#                 form_explainer_kwargs
#             )

#             # update kwargs on change in user input value
#             def update_current_kwargs(current_kwargs, key, value):
#                 current_kwargs.append((key, value))
#                 return current_kwargs, key, value

#             for form in form_explainer_kwargs:
#                 form.change(
#                     update_current_kwargs,
#                     inputs=[current_kwargs, gr.State(form.label), form],
#                     outputs=[current_kwargs, gr.State(form.label), form],
#                 )


#             # add explainer
#             btn_add_explainer = gr.Button('Add Explainer')
#             def add_explainer(explainers, new_explainer_nm, current_kwargs):
#                 kwargs = {}
#                 for key, value in reversed(current_kwargs):
#                     if kwargs.get(key) is None:
#                         kwargs[key] = value
#                 data = {
#                     'explainer_nm': new_explainer_nm,
#                     'kwargs': kwargs,
#                 }
#                 explainers.append(data)
#                 return explainers, new_explainer_nm, current_kwargs
#             btn_add_explainer.click(
#                 fn=add_explainer,
#                 inputs=[explainers, new_explainer_nm, current_kwargs],
#                 outputs=[explainers, new_explainer_nm, current_kwargs],
#             )

#         # render the list of explainers added
#         with gr.Column():
#             @gr.render(inputs=explainers)
#             def render_explainers(ls):
#                 gr.Markdown(f"### Added")
#                 for data in ls:
#                     gr.Textbox(f"{data['explainer_nm']}({', '.join([k+'='+str(v) for k, v in data['kwargs'].items()])})")

#         # TODO: submit explainers
#         # TODO: input form and submit inputs
#         # TODO: metric form and submit metrics
#         # TODO: run experiment by expr.run_batch and response results

# demo.launch(share=True)
