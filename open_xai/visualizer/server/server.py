from pnpxai.utils import Singleton
from pnpxai.visualizer.backend.app import create_app
from typing import Optional, List


class Server(metaclass=Singleton):
    def __init__(self):
        self.projects = []

    def register(self, project):
        self.projects.append(project)

    def unregister_by_name(self, name: str):
        for i, project in enumerate(self.projects):
            if project.name == name:
                del self.projects[i]
        raise ValueError(f"Project with name {name} not found")

    def unregister(self, project):
        self.projects.remove(project)

    def serve(self, projects: Optional[List] = None, **kwargs):
        projects = projects or self.projects
        if len(projects) == 0:
            raise ValueError("No projects to serve")

        projects_map = {project.name: project for project in projects}
        app = create_app(projects_map)
        app.run(**kwargs)
