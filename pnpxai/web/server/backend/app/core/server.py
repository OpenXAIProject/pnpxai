from flask import g
from pnpxai.web.server.server import Server


def set_server(server: Server):
    g.mgr_server = server


def get_server():
    if 'mgr_server' in g:
        return g.mgr_server
    return None
