from pnpxai.visualizer.backend.app.core.init_modules import get_projects


class ProjectService:
    @classmethod
    def get_all(cls):
        return get_projects()

    @classmethod
    def get_by_id(cls, name):
        projects = ProjectService.get_all()
        if projects is not None:
            return projects.get(name)
        return None
    
    @classmethod
    def get_experiments_with_names(cls, project):
        experiments = []
        for name, experiment in project.experiments.items():
            experiment.name = name
            experiments.append(experiment)
        return experiments
