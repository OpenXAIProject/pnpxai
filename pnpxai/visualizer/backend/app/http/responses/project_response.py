from pnpxai.visualizer.backend.app.http.responses.experiment_response import ExperimentResponse
from pnpxai.visualizer.backend.app.domain.project.project_service import ProjectService
from pnpxai.visualizer.backend.app.core.constants import APIItems
from pnpxai.visualizer.backend.app.core.generics import Response


class ProjectSchema(Response):
    @classmethod
    def to_dict(cls, project):
        experiments = ProjectService.get_experiments_with_names(project)

        return {
            APIItems.ID.value: project.name,
            APIItems.EXPERIMENTS.value: ExperimentResponse.dump(experiments, many=True)
        }
