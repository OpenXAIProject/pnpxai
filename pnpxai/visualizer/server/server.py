from pnpxai.utils import Singleton
from pnpxai.visualizer.backend.app import create_app
from typing import Optional, List
from werkzeug.serving import run_simple


class Server(metaclass=Singleton):
    def __init__(self):
        self._projects = []

    def register(self, project):
        self._projects.append(project)

    def reset(self):
        self._projects = []

    def unregister_by_name(self, name: str):
        self._projects = [
            project for project in self._projects if project.name != name]

    def unregister(self, project):
        self._projects = [
            _project for _project in self._projects if _project != project]

    def get_projects_map(self, projects: Optional[List] = None):
        projects = projects or self._projects or []
        projects_map = {project.name: project for project in projects}
        return projects_map
    
    def create_app(self, projects: Optional[List] = None, **kwargs):
        projects_map = self.get_projects_map(projects)
        if len(projects_map) == 0:
            raise ValueError("No projects to serve")

        app = create_app(projects_map)
        return app

    def serve(self, projects: Optional[List] = None, **kwargs):
        app = self.create_app(projects, **kwargs)
        app.run(**kwargs)

    def serve_ipynb(self, projects: Optional[List] = None, **kwargs):
        app = self.create_app(projects, **kwargs)
        run_simple('0.0.0.0', 5001, app)
