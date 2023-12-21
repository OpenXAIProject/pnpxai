import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pnpxai.utils import class_to_string
from pnpxai.evaluator._evaluator import weigh_metrics
from pnpxai.visualizer.backend.app.core.generics import Response
from pnpxai.visualizer.backend.app.core.constants import APIItems
from pnpxai.visualizer.backend.app.domain.experiment import ExperimentService


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


class ExperimentRunsResponse(Response):
    @classmethod
    def format_outputs_for_visualization(cls, outputs: Tensor, n_outputs: int = 3):
        if outputs is None:
            return outputs
        return outputs.argsort(descending=True)[:n_outputs].tolist()

    @classmethod
    def to_dict(cls, experiment):
        inputs = ExperimentService.get_task_formatted_inputs(
            experiment, experiment.get_inputs_flattened())
        targets = experiment.get_targets_flattened()
        outputs = [
            cls.format_outputs_for_visualization(output)
            for output in experiment.get_outputs_flattened()
        ]

        formatted = [
            {
                APIItems.INPUT.value: datum.to_json(),
                APIItems.TARGET.value: target,
                APIItems.OUTPUTS.value: output,
                APIItems.EXPLANATIONS.value: [],
            }
            for (datum, target, output) in zip(inputs, targets, outputs)
        ]

        explainers = experiment.get_current_explainers()
        metrics = experiment.get_current_metrics()
        evaluations = experiment.get_evaluations_flattened()
        visualizations = experiment.get_visualizations_flattened()

        for explainer, explainer_visualizations, explainer_evaluations in zip(explainers, visualizations, evaluations):
            for idx, visualization in enumerate(explainer_visualizations):
                formatted_evaluations = {
                    class_to_string(metric): metrics_evaluations[idx]
                    for metric, metrics_evaluations in zip(metrics, explainer_evaluations)
                }
                
                formatted[idx][APIItems.EXPLANATIONS.value].append({
                    APIItems.EXPLAINER.value: class_to_string(explainer.explainer),
                    APIItems.DATA.value: visualization.to_json() if visualization is not None else None,
                    APIItems.EVALUATION.value: formatted_evaluations,
                    APIItems.WEIGHTED_SCORE.value: weigh_metrics(formatted_evaluations),
                })

        for datum in formatted:
            datum[APIItems.EXPLANATIONS.value] = sorted(
                datum[APIItems.EXPLANATIONS.value],
                key=lambda x: x[APIItems.WEIGHTED_SCORE.value]
            )

        return formatted
