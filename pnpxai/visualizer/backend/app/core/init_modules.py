from flask import g
from flask_cors import CORS
from dataclasses import dataclass
from pnpxai.visualizer.backend.app.core.generics import JSONEncoder
from pnpxai.visualizer.backend.app.domain.project.project_progress_logger import ProjectProgressLogger


@dataclass
class InitConfig():
    projects: dict


_projects = None
_project_experiment_logger = None


def init_projects(projects: dict):
    global _projects
    global _project_experiment_logger
    _projects = projects
    _project_experiment_logger = ProjectProgressLogger()
    _project_experiment_logger.subscribe(projects)


def get_projects():
    global _projects
    return _projects


def get_project_experiment_logger():
    global _project_experiment_logger
    return _project_experiment_logger


def init_cors(app):
    CORS(app, resources={r'/*': {"origins": '*'}})
    app.config['CORS_HEADER'] = 'Content-Type'


def init_json_encoder(app):
    app.config["RESTX_JSON"] = {"cls": JSONEncoder}


def init(app, config: InitConfig):
    init_cors(app)
    init_json_encoder(app)
    init_projects(config.projects)
