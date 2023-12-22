from torch import Tensor
from plotly import express as px

from pnpxai.visualizer.backend.app.domain.experiment.task_visualizers._visualizer import BaseVisualizer


class ImageVisualizer(BaseVisualizer):
    @classmethod
    def format_inputs(cls, inputs, visualizer=None):
        formatted = []
        for datum in inputs:
            if datum is None:
                formatted.append(None)
                continue
            datum: Tensor = datum.cpu()

            if visualizer is not None:
                datum = visualizer(datum)

            fig = px.imshow(datum)
            formatted.append(fig)

        return formatted

    @classmethod
    def format_targets(cls, targets, visualizer=None):
        if visualizer is not None:
            targets = [
                visualizer(target) if target is not None else None
                for target in targets
            ]

        return targets

    @classmethod
    def format_outputs(cls, outputs, visualizer=None, n_outputs=3):
        formatted = []
        for output in outputs:
            if output is None:
                formatted.append(None)
                continue
            output: Tensor = output.cpu()
            top_outputs = output.argsort(descending=True)[:n_outputs]

            if visualizer is not None:
                top_outputs = [visualizer(output) for output in top_outputs]

            formatted.append(top_outputs)

        return formatted
