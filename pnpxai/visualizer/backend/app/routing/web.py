import os
from flask import Blueprint, send_from_directory, current_app

web_bp = Blueprint('web', __name__)


@web_bp.route('/', defaults={'path': ''})
@web_bp.route('/<path:path>')
def serve(path):
    print(current_app.static_folder)
    if path and os.path.exists(current_app.static_folder + '/' + path):
        return send_from_directory(current_app.static_folder, path)

    return send_from_directory(current_app.static_folder, 'index.html')
