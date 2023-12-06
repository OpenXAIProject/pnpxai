from flask import abort

from pnpxai.visualizer.backend.app.core.generics import Controller
from pnpxai.visualizer.backend.app.domain.project import ProjectService
from pnpxai.visualizer.backend.app.http.responses.project_response import ProjectSchema


class ProjectListController(Controller):
    def get(self):
        projects = list(ProjectService.get_all().values())
        return self.response(data=ProjectSchema.dump(projects, many=True))


class ProjectController(Controller):

    pass
