from flask import Blueprint
from flask_restx import Api
from pnpxai.visualizer.backend.app.http.controllers.project_controller import (
    ProjectListController,
)
from pnpxai.visualizer.backend.app.http.controllers.experiment_controller import (
    ExperimentController,
    ExperimentInputsController,
)
from pnpxai.visualizer.backend.app.http.controllers.model_controller import (
    ModelController
)

api_bp = Blueprint('/api', __name__, url_prefix='/api')
api = Api(api_bp)

routes = {
    '/projects/': ProjectListController,
    '/projects/<project_id>/experiments/<experiment_id>/': ExperimentController,
    '/projects/<project_id>/experiments/<experiment_id>/inputs/': ExperimentInputsController,
    '/projects/<project_id>/models/': ModelController
}

for url_rule, action in routes.items():
    api.add_resource(action, url_rule)
