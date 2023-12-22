from abc import abstractclassmethod


class BaseVisualizer:
    @abstractclassmethod
    def format_inputs(cls, inputs, visualizer=None):
        pass

    @abstractclassmethod
    def format_targets(cls, targets, visualizer=None):
        pass

    @abstractclassmethod
    def format_outputs(cls, outputs, visualizer=None, n_outputs: int = 3):
        pass


class DefaultVisualizer(BaseVisualizer):
    @classmethod
    def format_inputs(cls, inputs, visualizer=None):
        if visualizer is not None:
            inputs = [visualizer(datum) for datum in inputs]
        return inputs

    @classmethod
    def format_targets(cls, targets, visualizer=None):
        if visualizer is not None:
            targets = [visualizer(datum) for datum in targets]
        return targets

    @classmethod
    def format_outputs(cls, outputs, visualizer=None, n_outputs: int = 3):
        if visualizer is not None:
            outputs = [visualizer(datum) for datum in outputs[:n_outputs]]
        return outputs
