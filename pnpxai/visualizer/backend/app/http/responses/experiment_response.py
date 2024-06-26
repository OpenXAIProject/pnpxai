from torch import Tensor

from pnpxai.utils import class_to_string
from pnpxai.visualizer.backend.app.core.generics import Response
from pnpxai.visualizer.backend.app.core.constants import APIItems
from pnpxai.visualizer.backend.app.domain.experiment.experiment_service import ExperimentService


class ExperimentResponse(Response):
    @classmethod
    def format_classes_by_name(cls, values: list):
        return [
            {
                APIItems.ID.value: idx,
                APIItems.NAME.value: class_to_string(value),
            }
            for idx, value in enumerate(values)
        ]

    @classmethod
    def to_dict(cls, experiment):
        all_explainers = [
            explainer.explainer for explainer in experiment.all_explainers]
        explainers = cls.format_classes_by_name(all_explainers)
        metrics = cls.format_classes_by_name(experiment.all_metrics)

        fields = {
            APIItems.EXPLAINERS.value: explainers,
            APIItems.METRICS.value: metrics,
        }
        if hasattr(experiment, 'name'):
            fields[APIItems.NAME.value] = experiment.name

        return fields


class ExperimentInputsResponse(Response):
    @classmethod
    def to_dict(cls, figure):
        return figure.to_json()

class ExperimentStatusResponse(Response):
    @classmethod
    def to_dict(cls, logger):
        return {
            APIItems.PROGRESS.value: logger.progress,
            APIItems.MESSAGE.value: logger.message,
        }

class ExperimentRunsResponse(Response):
    @classmethod
    def to_dict(cls, experiment):
        inputs = ExperimentService.get_task_formatted_inputs(
            experiment, experiment.get_inputs_flattened())
        n_inputs = len(inputs)

        default_targets = [None] * n_inputs
        targets = experiment.get_targets_flattened()
        targets = ExperimentService.get_task_formatted_targets(
            experiment, targets) if targets is not None else default_targets

        default_outputs = [None, None] * n_inputs
        outputs = experiment.get_outputs_flattened()
        outputs = ExperimentService.get_task_formatted_outputs(
            experiment, outputs) if outputs is not None else default_outputs

        formatted = [
            {
                APIItems.INPUT.value: datum.to_json(),
                APIItems.TARGET.value: target,
                APIItems.OUTPUTS.value: [{
                    APIItems.KEY.value: output_id,
                    APIItems.VALUE.value: output
                } for output_id, output in outputs],
                APIItems.EXPLANATIONS.value: list(),
            }
            for (datum, target, outputs) in zip(inputs, targets, outputs)
        ]

        explainers = experiment.get_current_explainers()
        visualizations = experiment.get_visualizations_flattened()
        metrics = experiment.get_current_metrics()
        evaluations = experiment.get_evaluations_flattened()
        ranks = experiment.get_explainers_ranks()

        for explainer_id, (explainer, explainer_visualizations) in enumerate(zip(explainers, visualizations)):
            n_entries = len(explainer_visualizations)
            explainer_evaluations = evaluations[explainer_id] if evaluations is not None else\
                [None] * n_entries
            explainer_ranks = ranks[explainer_id] if ranks is not None else \
                [None] * n_entries

            for idx in range(n_entries):
                formatted_evaluations = {
                    class_to_string(metric): metrics_evaluations[idx]
                    for metric, metrics_evaluations in zip(metrics, explainer_evaluations)
                }
                visualization = explainer_visualizations[idx]
                explainer_rank = explainer_ranks[idx] if explainer_ranks is not None else None

                formatted[idx][APIItems.EXPLANATIONS.value].append({
                    APIItems.EXPLAINER.value: class_to_string(explainer.explainer),
                    APIItems.DATA.value: visualization.to_json() if visualization is not None else None,
                    APIItems.EVALUATION.value: formatted_evaluations,
                    APIItems.RANK.value: explainer_rank,
                })

        return formatted
