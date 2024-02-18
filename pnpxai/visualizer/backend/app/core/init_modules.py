from werkzeug.exceptions import HTTPException
from flask import g, Flask
from flask_cors import CORS
from dataclasses import dataclass
from pnpxai.visualizer.backend.app.core.generics import JSONEncoder, Controller
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


def handle_system_errors(e: Exception):
    if isinstance(e, HTTPException):
        return e

    return Controller.response(
        code=500,
        message=getattr(e, 'message', 'Internal Server Error'),
        errors=Controller.format_errors([e])
    )


def init(app: Flask, config: InitConfig):
    init_cors(app)
    init_json_encoder(app)
    init_projects(config.projects)

    app.register_error_handler(Exception, handle_system_errors)
