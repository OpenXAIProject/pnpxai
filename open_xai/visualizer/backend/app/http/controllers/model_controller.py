from flask import abort

from pnpxai.visualizer.backend.app.core.generics import Controller
from pnpxai.visualizer.backend.app.domain.project import ProjectService
from pnpxai.visualizer.backend.app.domain.experiment import ModelService
from pnpxai.visualizer.backend.app.http.responses.model_response import ModelSchema


class ModelController(Controller):
    def get(self, project_id: str):
        project = ProjectService.get_by_id(project_id)
        
        if project is None:
            abort(404)

        models = ModelService.get_all(project)
        return self.response(data=ModelSchema.dump(models, many=True))
