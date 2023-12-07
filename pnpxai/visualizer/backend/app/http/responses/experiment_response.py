import torch
from torch.utils.data import DataLoader

from pnpxai.visualizer.backend.app.core.generics import Response
from pnpxai.visualizer.backend.app.core.constants import APIItems
from pnpxai.visualizer.backend.app.domain.experiment import ExperimentService


class ExperimentResponse(Response):
    @classmethod
    def format_explainers(cls, explainers: list):
        return [
            {
                APIItems.ID.value: idx,
                APIItems.NAME.value: explainer.__name__
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


class ExperimentRunsResponse(Response):
    @classmethod
    def format_run_inputs(cls, experiment):
        run = next(iter(experiment.runs))
        if run is None:
            return []

        inputs = run.inputs
        if experiment.is_batched:
            in_shape = list(inputs.shape)
            print(in_shape)
            inputs = inputs.reshape(-1, *in_shape[2:])
        inputs = ExperimentService.get_task_formatted_inputs(
            experiment, inputs)

        return inputs

    @classmethod
    def to_dict(cls, experiment):
        inputs = cls.format_run_inputs(experiment)
        formatted = [
            {
                APIItems.INPUT.value: datum,
                APIItems.VISUALIZATIONS.value: [],
            }
            for datum in inputs
        ]

        is_batched = experiment.is_batched

        for run in experiment.runs:
            run_name = run.explainer.explainer.__class__.__name__
            run_visualizations = run.explainer.visualize()
            if is_batched:
                run_visualizations = torch.concat(run_visualizations, dim=0)
            for idx, visualization in enumerate(run_visualizations):
                formatted[idx][APIItems.VISUALIZATIONS.value].append({
                    APIItems.EXPLAINER.value: run_name,
                    APIItems.DATA.value: visualization,
                })

        return formatted
