from flask import g
from flask_cors import CORS
from dataclasses import dataclass
from pnpxai.visualizer.backend.app.core.generics import JSONEncoder


@dataclass
class InitConfig():
    projects: dict


_projects = None


def init_projects(projects: dict):
    global _projects
    _projects = projects


def get_projects():
    global _projects
    return _projects


def init_cors(app):
    CORS(app, resources={r'/*': {"origins": '*'}})
    app.config['CORS_HEADER'] = 'Content-Type'


def init_json_encoder(app):
    app.config["RESTX_JSON"] = {"cls": JSONEncoder}


def init(app, config: InitConfig):
    init_cors(app)
    init_json_encoder(app)
    init_projects(config.projects)
