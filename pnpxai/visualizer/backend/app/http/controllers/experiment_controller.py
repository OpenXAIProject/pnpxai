from pnpxai.visualizer.backend.app.core.generics import Controller
from pnpxai.visualizer.backend.app.domain.project.project_service import ProjectService
from pnpxai.visualizer.backend.app.domain.experiment.experiment_service import ExperimentService


class ExperimentListController(Controller):
    pass


class ExperimentController(Controller):
    pass


class ExperimentInputsController(Controller):
    def get(self, project_id: str, experiment_id: str):
        experiment = ProjectService.get_experiment_by_id(
            project_id, experiment_id)

        inputs = ExperimentService.get_task_formatted_inputs(experiment)

        return self.response(data=inputs)
