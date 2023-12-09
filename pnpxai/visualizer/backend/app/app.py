from pnpxai.visualizer.backend.app.core.init_modules import init, InitConfig
from flask import Flask, send_from_directory
import os

def create_app(projects: dict):
    # Correct the path to the build directory
    frontend_build_path = os.path.join(os.path.dirname(__file__), '../../../frontend/build')

    application = Flask(__name__, static_folder=frontend_build_path, static_url_path='/')
    # application.config.from_object(Config)
    init(application, InitConfig(projects=projects))

    from pnpxai.visualizer.backend.app.routing.api import api_bp
    application.register_blueprint(api_bp)

    # Serve static files and index.html for routes not in api_bp
    @application.route('/', defaults={'path': ''})
    @application.route('/<path:path>')
    def serve(path):
        if path and os.path.exists(application.static_folder + '/' + path):
            return send_from_directory(application.static_folder, path)
        else:
            return send_from_directory(application.static_folder, 'index.html')

    return application
