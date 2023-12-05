from flask import Blueprint
from flask_restx import Api
from pnpxai.visualizer.backend.app.http.controllers.project_controller import (
    ProjectListController,
    ProjectController,
)
from pnpxai.visualizer.backend.app.http.controllers.experiment_controller import (
    ExperimentListController,
    ExperimentController,
    ExperimentInputsController,
)

api_bp = Blueprint('/api', __name__, url_prefix='/api')
api = Api(api_bp)

routes = {
    '/projects/': ProjectListController,
    '/projects/<project_id>/': ProjectController,
    '/projects/<project_id>/experiments/': ExperimentListController,
    '/projects/<project_id>/experiments/<experiment_id>/': ExperimentController,
    '/projects/<project_id>/experiments/<experiment_id>/inputs/': ExperimentInputsController
}

for url_rule, action in routes.items():
    api.add_resource(action, url_rule)
