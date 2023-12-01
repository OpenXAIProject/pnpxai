from pnpxai.visualizer.backend.app.core.server import init_mgr_client
from pnpxai.visualizer import config
from flask import Flask, g
from flask_cors import CORS


def create_app(address: str = config.MGR_DFT_ADDRESS, port: int = config.MGR_DFT_PORT, password: str = config.MGR_DFT_PASSWORD):
    application = Flask(__name__)
    # application.config.from_object(Config)

    CORS(application, resources={r'/*': {"origins": '*'}})
    application.config['CORS_HEADER'] = 'Content-Type'

    # from src.app.routing.api import api_bp
    # application.register_blueprint(api_bp)
    init_mgr_client(application, address, port, password)

    return application
