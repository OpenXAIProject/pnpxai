from flask import abort, request

from pnpxai.visualizer.backend.app.core.generics import Controller
from pnpxai.visualizer.backend.app.domain.project import ProjectService
from pnpxai.visualizer.backend.app.domain.experiment import ExperimentService
from pnpxai.visualizer.backend.app.http.responses.experiment_response import ExperimentRunsResponse, ExperimentInputsResponse


class ExperimentController(Controller):
    def get(self, project_id: str, experiment_id: str):
        experiment = ProjectService.get_experiment_by_id(
            project_id, experiment_id)

        if experiment is None:
            abort(404)

        if not experiment.has_explanations:
            return self.response(data=[])

        return self.response(data=ExperimentRunsResponse.dump(experiment))

    def put(self, project_id: str, experiment_id: str):
        experiment = ProjectService.get_experiment_by_id(
            project_id, experiment_id)

        if experiment is None:
            abort(404)

        data = request.get_json() or {}
        inputs = data.get('inputs', None)
        explainers = data.get('explainers', None)
        metrics = data.get('metrics', None)

        experiment = ExperimentService.run(
            experiment, inputs, explainers, metrics)

        return self.response(data=ExperimentRunsResponse.dump(experiment))


class ExperimentInputsController(Controller):
    def get(self, project_id: str, experiment_id: str):
        experiment = ProjectService.get_experiment_by_id(
            project_id, experiment_id)

        if experiment is None:
            abort(404)

        figures = ExperimentService.get_task_formatted_inputs(experiment, experiment.get_all_inputs_flattened())
        return self.response(data=ExperimentInputsResponse.dump(figures, many=True))
