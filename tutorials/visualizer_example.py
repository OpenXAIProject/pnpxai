from pnpxai.visualizer.visualizer import Visualizer
from pnpxai import AutoExplanation, Modality
import torch
import numpy as np
import os

# ------------------------------------------------------------------------------#
# ----------------------------------- setup ------------------------------------#
# ------------------------------------------------------------------------------#

from helpers import load_model_and_dataloader_for_tutorial, denormalize_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, loader, transform = load_model_and_dataloader_for_tutorial("image", device)
sample_batch = next(iter(loader))
modality = Modality(
    dtype=sample_batch[0].dtype,
    ndims=sample_batch[0].dim(),
    pooling_dim=1,
)


# ------------------------------------------------------------------------------#
# ---------------------------- auto experiment ---------------------------------#
# ------------------------------------------------------------------------------#


expr = AutoExplanation(
    model=model,
    data=loader,
    modality=modality,
    target_input_keys=[0],
    target_class_extractor=lambda outputs: outputs.argmax(-1),
    label_key=-1,
    target_labels=True,
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
visualizer.launch(share=True)
