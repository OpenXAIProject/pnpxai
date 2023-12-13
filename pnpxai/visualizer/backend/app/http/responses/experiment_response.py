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
                APIItems.EXPLANATIONS.value: [],
            }
            for datum in inputs
        ]

        for run in experiment.runs:
            run_name = class_to_string(run.explainer.explainer)
            run_visualizations = run.get_flattened_visualizations(
                experiment.task
            )
            evaluations = run.flattened_evaluations or [
                None for _ in range(len(run_visualizations))
            ]
            weighted_evaluations = run.flattened_weighted_evaluations

            for idx, (visualization, evaluation, weighted_score) in enumerate(zip(run_visualizations, evaluations, weighted_evaluations)):
                formatted[idx][APIItems.EXPLANATIONS.value].append({
                    APIItems.EXPLAINER.value: run_name,
                    APIItems.DATA.value: visualization.to_json() if visualization is not None else None,
                    APIItems.EVALUATION.value: evaluation,
                    APIItems.WEIGHTED_SCORE.value: weighted_score,
                })

        for datum in formatted:
            datum[APIItems.EXPLANATIONS.value] = sorted(
                datum[APIItems.EXPLANATIONS.value],
                key=lambda x: x[APIItems.WEIGHTED_SCORE.value],
                reverse=True,
            )

        return formatted
