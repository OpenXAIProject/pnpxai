from typing import Dict, Optional
import threading
from multiprocessing import Lock
from multiprocessing.managers import BaseManager

from pnpxai.core.project import Project
from pnpxai.visualizer import config


class Server():
    projects: Dict[str, Project] = {}

    def __init__(
        self,
        address: Optional[str] = None,
        port: Optional[int] = None
    ):
        address = address or config.MGR_DFT_ADDRESS
        port = port or config.MGR_DFT_PORT

        self.__init_manager(address, port)
        self.__init_lock()

    def __init_manager(self, address: str, port: int):
        self.manager = BaseManager((address, port))
        self.manager.register(
            config.MGR_FC_GET_ALL_PRJS, self.get_all_projects
        )
        self.manager.register(config.MGR_FC_GET_PRJ, self.get_project)
        self.manager.register(config.MGR_FC_SYNC_PRJ, self.sync_project)
        self.manager.register(config.MGR_FC_SHUTDOWN, self.shutdown)

    def __init_lock(self):
        self.lock = Lock()

    def get_all_projects(self) -> Dict[str, Project]:
        return self.projects

    def get_project(self, key: str) -> Optional[Project]:
        return self.projects.get(key, None)

    def sync_project(self, key: str, project: Project):
        projects = self.projects
        with self.lock:
            projects[key] = project
            self.projects = projects

    def shutdown(self):
        server = self.manager.get_server()

        def _shutdown():
            server.stop_event().set()
            self.manager.shutdown()

        threading.Timer(1, _shutdown).start()

    def serve_forever(self):
        server = self.manager.get_server()
        server.serve_forever()
