from flask import abort, request

from pnpxai.visualizer.backend.app.core.generics import Controller
from pnpxai.visualizer.backend.app.domain.project import ProjectService
from pnpxai.visualizer.backend.app.domain.experiment import ExperimentService
from pnpxai.visualizer.backend.app.http.responses.experiment_response import ExperimentRunsResponse


class ExperimentListController(Controller):
    pass


class ExperimentController(Controller):
    def put(self, project_id: str, experiment_id: str):
        experiment = ProjectService.get_experiment_by_id(
            project_id, experiment_id)
        print(experiment)
        if experiment is None:
            abort(404)

        data = request.get_json() or {}
        inputs = data.get('inputs', None)
        explainers = data.get('explainers', None)

        experiment = ExperimentService.run(experiment, inputs, explainers)

        return self.response(data=ExperimentRunsResponse.dump(experiment))


class ExperimentInputsController(Controller):
    def get(self, project_id: str, experiment_id: str):
        experiment = ProjectService.get_experiment_by_id(
            project_id, experiment_id)
        
        if experiment is None:
            abort(404)

        inputs = ExperimentService.get_task_formatted_inputs(experiment)

        return self.response(data=inputs)
