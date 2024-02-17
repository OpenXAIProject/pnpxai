from pnpxai.visualizer.backend.app.domain.experiment.experiment_progress_logger import ExperimentProgressLogger


class ProjectProgressLogger:
    def __init__(self):
        self._project_experiment_logger_map = {}

    def subscribe(self, projects: dict):
        for project_name, project in projects.items():
            for experiment_name, experiment in project.experiments.items():
                logger = ExperimentProgressLogger()
                experiment.subscribe(logger.log)
                self._project_experiment_logger_map[self.get_project_experiment_key(
                    project_name, experiment_name)] = logger

    @classmethod
    def get_project_experiment_key(cls, project: str, experiment: str):
        return f"Project:{project}->Experiment:{experiment}"

    def get_experiment_logger(self, project: str, experiment: str):
        key = self.get_project_experiment_key(project, experiment)
        return self._project_experiment_logger_map.get(key)
