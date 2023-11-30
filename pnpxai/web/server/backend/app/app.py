from pnpxai.web.server.server import Server
from pnpxai.web.server.backend.app.core.server import set_server
from flask import Flask, g
from flask_cors import CORS


def create_app(server: Server):
    application = Flask(__name__)
    # application.config.from_object(Config)

    CORS(application, resources={r'/*': {"origins": '*'}})
    application.config['CORS_HEADER'] = 'Content-Type'

    # from src.app.routing.api import api_bp
    # application.register_blueprint(api_bp)
    set_server(server)

    return application
