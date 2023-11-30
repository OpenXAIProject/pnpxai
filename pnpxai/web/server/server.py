from typing import Dict, Optional
from multiprocessing import Lock
from multiprocessing.managers import BaseManager

from pnpxai.core.project import Project
from pnpxai.web import config


class Server():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Server, cls).__new__(cls)
        return cls.instance

    projects: Dict[str, Project] = {}

    def __init__(
        self,
        address: str = config.MGR_DFT_ADDRESS,
        port: int = config.MGR_DFT_PORT,
        password: bytearray = config.MGR_DFT_PASSWORD,
    ):
        self.__init_manager(address, port, password)
        self.__init_lock()

    def __init_manager(self, address: str, port: int, password: bytearray):
        self.manager = BaseManager((address, port), password)
        self.manager.register(
            config.MGR_FC_GET_ALL_PRJS, self.get_all_projects
        )
        self.manager.register(config.MGR_FC_GET_PRJ, self.get_project)
        self.manager.register(config.MGR_FC_SYNC_PRJ, self.sync_project)

    def __init_lock(self):
        self.lock = Lock()

    def get_all_projects(self) -> Dict[str, Project]:
        return self.projects

    def get_project(self, key: str) -> Optional[Project]:
        return self.projects.get(key, None)

    def sync_project(self, key: str, project: Project):
        with self.lock:
            self.projects[key] = project

    def serve_forever(self):
        server = self.manager.get_server()
        server.serve_forever()
