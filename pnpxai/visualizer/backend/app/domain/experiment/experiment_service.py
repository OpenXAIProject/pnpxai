from typing import Optional, Sequence
from pnpxai.visualizer.backend.app.domain.experiment.task_visualizers.visualizer_factory import VisualizerFactory


class ExperimentService:
    @classmethod
    def get_task_formatted_inputs(cls, experiment, inputs=None):
        inputs = inputs \
            if inputs is not None else \
            experiment.get_inputs_flattened()

        return VisualizerFactory.get_visualizer(experiment).format_inputs(inputs, experiment.input_visualizer)

    @classmethod
    def get_task_formatted_targets(cls, experiment, targets=None):
        targets = targets \
            if targets is not None else \
            experiment.get_targets_flattened()

        return VisualizerFactory.get_visualizer(experiment).format_targets(targets, experiment.target_visualizer)

    @classmethod
    def get_task_formatted_outputs(cls, experiment, outputs=None, n_outputs=3):
        outputs = outputs \
            if outputs is not None else \
            experiment.get_outputs_flattened()

        return VisualizerFactory.get_visualizer(experiment).format_outputs(outputs, experiment.target_visualizer, n_outputs)

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
