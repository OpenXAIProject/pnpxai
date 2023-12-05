from pnpxai.visualizer.backend.app.core.generics import Response
from pnpxai.visualizer.backend.app.core.constants import APIItems


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
