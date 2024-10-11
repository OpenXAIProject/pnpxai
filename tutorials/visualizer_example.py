from pnpxai.visualizer.visualizer import Visualizer
from pnpxai import AutoExplanationForImageClassification
import torch
import numpy as np
import os

# ------------------------------------------------------------------------------#
# ----------------------------------- setup ------------------------------------#
# ------------------------------------------------------------------------------#

from helpers import load_model_and_dataloader_for_tutorial, denormalize_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, loader, transform = load_model_and_dataloader_for_tutorial("image", device)


# ------------------------------------------------------------------------------#
# ---------------------------- auto experiment ---------------------------------#
# ------------------------------------------------------------------------------#


expr = AutoExplanationForImageClassification(
    model=model,
    data=loader,
    input_extractor=lambda batch: batch[0].to(device),
    label_extractor=lambda batch: batch[1].to(device),
    target_extractor=lambda outputs: outputs.argmax(-1).to(device),
    channel_dim=1,
)


# run_batch returns a dict of results
results = expr.run_batch(
    data_ids=[0, 1], explainer_id=1, postprocessor_id=0, metric_id=1
)

# Also, we can use experiment manager to browse results as follows
# get data
# a pair of single instance (input, label)
data = expr.manager.get_data_by_id(0)
# a batch of multiple instances (inputs, labels)
batch = expr.manager.batch_data_by_ids(data_ids=[0, 1])

# get explainer
explainer = expr.manager.get_explainer_by_id(0)

# get explanation
attr = expr.manager.get_explanation_by_id(data_id=0, explainer_id=1)
batched_attrs = expr.manager.batch_explanations_by_ids(
    data_ids=[0, 1], explainer_id=1
)  # batched explanations

# get postprocessor
postprocessor = expr.manager.get_postprocessor_by_id(0)

# postprocess
# Note that this work only for batched attrs
postprocessed = postprocessor(batched_attrs)

# get metric
metric = expr.manager.get_metric_by_id(0)

# get evaluation
evaluation = expr.manager.get_evaluation_by_id(
    data_id=0,
    explainer_id=1,
    postprocessor_id=0,
    metric_id=1,
)
evaluations = expr.manager.batch_evaluations_by_ids(  # batched evaluations
    data_ids=[0, 1],
    explainer_id=1,
    postprocessor_id=0,
    metric_id=1,
)


def input_visualizer(datum: torch.Tensor) -> np.ndarray:
    return denormalize_image(
        datum.detach().cpu(),
        mean=transform.mean,
        std=transform.std,
    )


os.environ["TMPDIR"] = "~/tmp"
visualizer = Visualizer(
    experiment=expr,
    input_visualizer=input_visualizer,
)
visualizer.launch()
