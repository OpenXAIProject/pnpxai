from torch import Tensor
from torch.utils.data import DataLoader
import plotly.express as px
from typing import Optional, Sequence


class ExperimentService:
    @classmethod
    def get_inputs_list(cls, experiment):
        data = experiment.data
        if isinstance(data, DataLoader):
            data = data.dataset

        data = list(map(experiment.input_extractor, data))

        return data

    @classmethod
    def get_task_formatted_inputs(cls, experiment, inputs=None):
        inputs = inputs \
            if inputs is not None else \
            cls.get_inputs_list(experiment)

        if experiment.is_image_task:
            inputs = cls._format_image_inputs(
                inputs, experiment.input_visualizer
            )

        return inputs

    @classmethod
    def _format_image_inputs(cls, inputs, visualizer=None):
        formatted = []
        for datum in inputs:
            datum: Tensor = datum.cpu()

            if visualizer is not None:
                datum = visualizer(datum)

            fig = px.imshow(datum)
            formatted.append(fig)

        return formatted

    @classmethod
    def run(
        cls,
        experiment,
        inputs: Optional[Sequence[int]] = None,
        explainers: Optional[Sequence[int]] = None,
        metrics: Optional[Sequence[int]] = None
    ):
        experiment.run(inputs, explainers, metrics)

        return experiment
