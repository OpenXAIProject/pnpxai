from pnpxai.visualizer.backend.app.core.init_modules import init, InitConfig
from flask import Flask
import os


def create_app(projects: dict):
    # Correct the path to the build directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the target directory
    frontend_build_path = os.path.normpath(
        os.path.join(current_script_dir, '../../frontend/build')
    )
    application = Flask(
        __name__,
        static_folder=frontend_build_path,
        static_url_path='/'
    )

    init(application, InitConfig(projects=projects))

    from pnpxai.visualizer.backend.app.routing.api import api_bp
    application.register_blueprint(api_bp)

    # Serve static files and index.html for routes not in api_bp
    from pnpxai.visualizer.backend.app.routing.web import web_bp
    application.register_blueprint(web_bp)

    return application
