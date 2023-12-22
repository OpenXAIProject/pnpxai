from typing import Type

from pnpxai.visualizer.backend.app.domain.experiment.task_visualizers.image_visualizer import ImageVisualizer
from pnpxai.visualizer.backend.app.domain.experiment.task_visualizers._visualizer import DefaultVisualizer, BaseVisualizer

class VisualizerFactory:
    @classmethod
    def get_visualizer(cls, experiment) -> Type[BaseVisualizer]:
        if experiment.is_image_task:
            return ImageVisualizer
        return DefaultVisualizer