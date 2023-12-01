from flask import g
from pnpxai.visualizer.proc_manager.client import Client


def init_mgr_client(app, address: str, port: int, password: str):
    with app.app_context():
        client = Client(address, port, password)
        g.mgr_client = None
        if client.connect_to_server():
            g.mgr_client = client


def get_mgr_client():
    if 'mgr_client' in g:
        return g.mgr_client
    return None
