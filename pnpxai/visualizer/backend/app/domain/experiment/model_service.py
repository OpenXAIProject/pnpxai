class ModelService:
    @classmethod
    def get_all(cls, project):
        return [
            experiment.model
            for experiment in project.experiments.values()
        ]
