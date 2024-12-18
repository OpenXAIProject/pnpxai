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
