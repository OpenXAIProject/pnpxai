from pnpxai.visualizer.backend.app.core.init_modules import init, InitConfig
from flask import Flask
import os


def create_app(projects: dict):
    # Correct the path to the build directory
    frontend_build_path = os.path.join(
        os.path.dirname(__file__), '../../../frontend/build'
    )

    application = Flask(
        __name__,
        static_folder=frontend_build_path,
        static_url_path='/'
    )
    # application.config.from_object(Config)
    init(application, InitConfig(projects=projects))

    from pnpxai.visualizer.backend.app.routing.api import api_bp
    application.register_blueprint(api_bp)

    # Serve static files and index.html for routes not in api_bp
    from pnpxai.visualizer.backend.app.routing.web import web_bp
    application.register_blueprint(web_bp)

    return application
