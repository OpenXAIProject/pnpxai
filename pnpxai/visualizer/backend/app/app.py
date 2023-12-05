from pnpxai.visualizer.backend.app.core.init_modules import init, InitConfig
from flask import Flask

def create_app(projects: dict):

    application = Flask(__name__)
    # application.config.from_object(Config)
    init(application, InitConfig(projects=projects))

    from pnpxai.visualizer.backend.app.routing.api import api_bp
    application.register_blueprint(api_bp)

    return application
