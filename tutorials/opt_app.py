import gradio as gr
import functools
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pnpxai import AutoExplanationForImageClassification

from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image


# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model, transform = get_torchvision_model('vit_b_16')
model.eval()
dataset = get_imagenet_dataset(transform, indices=range(1000))
loader = DataLoader(dataset, batch_size=4, shuffle=False)
expr = AutoExplanationForImageClassification(
    model=model.to(device),
    data=loader,
    input_extractor=lambda batch: batch[0].to(device),
    label_extractor=lambda batch: batch[-1].to(device),
    target_extractor=lambda outputs: outputs.argmax(-1).to(device),
    target_labels=False,  # target prediction if False
)


# ------------------------------------------------------------------------------#
# --------------------------------- client -------------------------------------#
# ------------------------------------------------------------------------------#

# run with default hp
def run_default(data_id, explainer_id, postprocessor_id, metric_id):
    # Then,
    # - model outputs,
    # - explanation before postprocessing,
    # - evaluation by the metric
    # are cached.
    return expr.run_batch(
        explainer_id=explainer_id,
        postprocessor_id=postprocessor_id,
        metric_id=metric_id,
        data_ids=[data_id],
    )

# get data


def get_input_image(data_id):
    batch = expr.manager.batch_data_by_ids(data_ids=[data_id])
    input_img = expr.input_extractor(batch)[0]
    return denormalize_image(
        input_img.detach().cpu(), mean=transform.mean, std=transform.std,
    )


def get_label(data_id):
    batch = expr.manager.batch_data_by_ids(data_ids=[data_id])
    label = expr.label_extractor(batch)[0].item()
    return dataset.dataset.idx_to_label(str(label))


def get_target(data_id):
    target = expr._get_targets(data_ids=[data_id])[0].item()
    return dataset.dataset.idx_to_label(str(target))


def get_prob(data_id):
    output = expr.manager.get_output_by_id(data_id=data_id)
    return output.softmax(-1).max().item()


def get_explainer(explainer_id):
    explainer = expr.manager.get_explainer_by_id(explainer_id)
    return str(explainer)


def get_postprocessor(postprocessor_id):
    return str(expr.manager.get_postprocessor_by_id(postprocessor_id))


def get_metric(metric_id):
    return str(expr.manager.get_metric_by_id(metric_id))


def get_explanation(data_id, explainer_id, postprocessor_id, cmap='YlGn'):
    attrs = expr.manager.batch_explanations_by_ids(
        data_ids=[data_id],
        explainer_id=explainer_id,
    )
    postprocessor = expr.manager.get_postprocessor_by_id(postprocessor_id)
    postprocessed = postprocessor(attrs)[0].detach().cpu().numpy()
    return coloring_heatmap(postprocessed, cmap=cmap)


def coloring_heatmap(postprocessed, cmap='YlGn'):
    cmap = plt.get_cmap(cmap)
    return cmap(postprocessed)


def get_evaluation(data_id, explainer_id, postprocessor_id, metric_id):
    return expr.manager.get_evaluation_by_id(
        data_id=data_id,
        explainer_id=explainer_id,
        postprocessor_id=postprocessor_id,
        metric_id=metric_id,
    )


def post_default_explanation(
    explainer_id,
    postprocessor_id,
    metric_id,
    cmap,
    data_id=None,
):
    run_default(data_id, explainer_id, postprocessor_id, metric_id)
    return (
        get_target(data_id),
        get_prob(data_id),
        get_explanation(data_id, explainer_id, postprocessor_id, cmap),
        get_evaluation(data_id, explainer_id, postprocessor_id, metric_id),
    )


N_TRIALS = 50
OPTIM_SEED = 42


def post_opt(
    explainer_id,
    metric_id,
    cmap,
    data_id=None
):
    optimized, objective, study = expr.optimize(
        data_ids=data_id,
        explainer_id=explainer_id,
        metric_id=metric_id,
        direction='maximize',
        sampler='tpe',
        n_trials=N_TRIALS,
        seed=OPTIM_SEED,
    )
    trials = [trial for trial in study.trials if trial.value is not None]
    trials = sorted(trials, key=lambda trial: trial.value)
    qsize = len(trials) // 4
    q_indices = [0, qsize*1, qsize*2, qsize*3, -1]

    # prepare inputs
    batch = expr.manager.batch_data_by_ids([data_id])
    inputs = expr.input_extractor(batch)
    targets = expr._get_targets([data_id])
    attrs = []
    explainers = []
    postprocessors = []
    values = []
    for q in q_indices:
        explainer, postprocessor = objective.load_from_optuna_params(
            trials[q].params)
        explainers.append(str(explainer))
        postprocessors.append(str(postprocessor))
        values.append(trials[q].value)
        postprocessed = postprocessor(explainer.attribute(inputs, targets))[0]
        attrs.append(coloring_heatmap(
            postprocessed.detach().cpu().numpy(), cmap=cmap))
    return tuple([*attrs, *explainers, *postprocessors, *values])


CMAPS = ['Reds', 'YlGn', 'coolwarm', 'viridis', 'jet']


# ------------------------------------------------------------------------------#
# ----------------------------------- app --------------------------------------#
# ------------------------------------------------------------------------------#


# Assume that data_id was selected through the vanilla user scenario.
data_id = 1

with gr.Blocks() as demo:
    # inputs
    input_img = gr.Image(
        value=get_input_image(data_id),
        label='Input Image',
    )
    input_label = gr.Textbox(
        value=get_label(data_id),
        label='Ground-truth Label',
    )
    explainer_id = gr.Dropdown(
        choices=[
            get_explainer(explainer_id)
            for explainer_id in range(len(expr.manager.explainers))
        ],
        type='index',
        label='Default Explainer',
    )
    postprocessor_id = gr.Dropdown(
        choices=[
            get_postprocessor(postprocessor_id)
            for postprocessor_id in range(len(expr.manager.postprocessors))
        ],
        type='index',
        label='Post Processor',
    )
    cmap = gr.Dropdown(
        choices=CMAPS,
        type='value',
        label='Color Map',
    )
    metric_id = gr.Dropdown(
        choices=[
            get_metric(metric_id)
            for metric_id in range(len(expr.manager.metrics))
        ],
        type='index',
        label='Metric',
    )

    btn_default_expl = gr.Button('Run')

    # outputs
    target = gr.Textbox(label='Prediction')
    prob = gr.Textbox(label='Prob. Score')
    expl = gr.Image(label='Explanation')
    ev = gr.Textbox(label='Evaluation')
    btn_default_expl.click(
        fn=functools.partial(post_default_explanation, data_id=data_id),
        inputs=[explainer_id, postprocessor_id, metric_id, cmap],
        outputs=[target, prob, expl, ev],
    )

    # optimize
    btn_opt = gr.Button('Optimize')
    with gr.Row():
        attr_worst = gr.Image(label='Worst')
        attr_q1 = gr.Image(label='Q1')
        attr_q2 = gr.Image(label='Q2')
        attr_q3 = gr.Image(label='Q3')
        attr_best = gr.Image(label='Best')
        attrs = [attr_worst, attr_q1, attr_q2, attr_q3, attr_best]

    with gr.Row():
        explainer_worst = gr.Textbox(label='Worst: params')
        explainer_q1 = gr.Textbox(label='Q1: params')
        explainer_q2 = gr.Textbox(label='Q2: params')
        explainer_q3 = gr.Textbox(label='Q3: params')
        explainer_best = gr.Textbox(label='Best: params')
        explainers = [
            explainer_worst,
            explainer_q1,
            explainer_q2,
            explainer_q3,
            explainer_best,
        ]

    with gr.Row():
        postprocessor_worst = gr.Textbox(label='Worst: postprocess')
        postprocessor_q1 = gr.Textbox(label='Q1: postprocess')
        postprocessor_q2 = gr.Textbox(label='Q2: postprocess')
        postprocessor_q3 = gr.Textbox(label='Q3: postprocess')
        postprocessor_best = gr.Textbox(label='Best: postprocess')
        postprocessors = [
            postprocessor_worst,
            postprocessor_q1,
            postprocessor_q2,
            postprocessor_q3,
            postprocessor_best,
        ]

    with gr.Row():
        values = [
            gr.Textbox(label=f'{k}: evaluation')
            for k in ['Worst', 'Q1', 'Q2', 'Q3', 'Best']
        ]

    btn_opt.click(
        fn=functools.partial(post_opt, data_id=data_id),
        inputs=[explainer_id, metric_id, cmap],
        outputs=[*attrs, *explainers, *postprocessors, *values],
    )

demo.launch(share=True)
