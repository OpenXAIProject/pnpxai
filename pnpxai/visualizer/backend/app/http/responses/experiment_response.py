import torch
from torch.utils.data import DataLoader

from pnpxai.utils import class_to_string
from pnpxai.visualizer.backend.app.core.generics import Response
from pnpxai.visualizer.backend.app.core.constants import APIItems
from pnpxai.visualizer.backend.app.domain.experiment import ExperimentService


class ExperimentResponse(Response):
    @classmethod
    def format_explainers(cls, explainers: list):
        return [
            {
                APIItems.ID.value: idx,
                APIItems.NAME.value: explainer.__name__,
            }
            for idx, explainer in enumerate(explainers)
        ]

    @classmethod
    def to_dict(cls, experiment):
        explainers = cls.format_explainers(experiment.available_explainers)

        fields = {
            APIItems.EXPLAINERS.value: explainers,
        }
        if hasattr(experiment, 'name'):
            fields[APIItems.NAME.value] = experiment.name

        return fields

class ExperimentInputsResponse(Response):
    @classmethod
    def to_dict(cls, figure):
        return figure.to_json()


class ExperimentRunsResponse(Response):
    @classmethod
    def format_run_inputs(cls, experiment):
        run = next(iter(experiment.runs))
        if run is None:
            return []

        inputs = [run.input_extractor(datum) for datum in run.data]
        if experiment.is_batched:
            inputs = list(torch.concat(inputs, dim=0))

        inputs = ExperimentService.get_task_formatted_inputs(
            experiment, inputs
        )

        return inputs

    @classmethod
    def to_dict(cls, experiment):
        inputs = cls.format_run_inputs(experiment)
        formatted = [
            {
                APIItems.INPUT.value: datum.to_json(),
                APIItems.VISUALIZATIONS.value: [],
            }
            for datum in inputs
        ]

        for run in experiment.runs:
            run_name = class_to_string(run.explainer.explainer)
            run_visualizations = run.visualize(experiment.task)
            run_visualizations = sum(run_visualizations, [])
            for idx, visualization in enumerate(run_visualizations):
                formatted[idx][APIItems.VISUALIZATIONS.value].append({
                    APIItems.EXPLAINER.value: run_name,
                    APIItems.DATA.value: visualization.to_json() if visualization is not None else None,
                })
        print(formatted)

        return formatted
