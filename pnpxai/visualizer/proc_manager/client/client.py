from multiprocessing.managers import BaseManager
import pathlib
import sys
import time
from typing import Tuple, Optional

from pnpxai.visualizer.utils import start_detached_process
from pnpxai.visualizer import config


class Client():
    def __init__(
        self,
        address: Optional[str] = None,
        port: Optional[int] = None,
    ):
        address = address if address is not None else config.MGR_DFT_ADDRESS
        port = port if port is not None else config.MGR_DFT_PORT

        self.manager = self.get_new_manager(address=address, port=port)

    def get_new_manager(self, address: str, port: int):
        manager = BaseManager((address, port))
        manager.register(config.MGR_FC_GET_ALL_PRJS)
        manager.register(config.MGR_FC_GET_PRJ)
        manager.register(config.MGR_FC_SYNC_PRJ)
        manager.register(config.MGR_FC_SHUTDOWN)
        return manager

    def _get_manager_credentials(self) -> Tuple[str, int]:
        return self.manager.address

    def _safe_connect_to_server(self):
        try:
            self.manager.connect()
            print("Connected to the server")
        except ConnectionRefusedError as e:
            return False
        return True

    def connect_to_server(self, n_tries: int = 20, delay: float = 0.1):
        for _ in range(n_tries):
            if self._safe_connect_to_server():
                return True
            time.sleep(delay)
        return False

    def start_server(self, address: str, port: int):
        proc_dir = pathlib.Path(__file__).parent.parent.resolve()
        proc_dir = proc_dir.joinpath('server')
        cmd = [
            sys.executable, f"{proc_dir}/run.py",
            "--address", address,
            "--port", str(port)
        ]
        start_detached_process(cmd)

    def start_web_server(self, address: str, port: int):
        proc_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
        proc_dir = proc_dir.joinpath('backend')
        cmd = [
            sys.executable, f"{proc_dir}/run.py",
            "--address", address,
            "--port", str(port)
        ]
        print("CMD: ", cmd)
        start_detached_process(cmd)

    def connect_to_or_start_server(self):
        if self.connect_to_server():
            return

        address, port = self._get_manager_credentials()
        self.start_server(address, port)
        self.connect_to_server()
        # self.start_web_server(address, port)

    def get_all_projects(self):
        return getattr(self.manager, config.MGR_FC_GET_ALL_PRJS)()

    def get_project(self, key: str):
        return getattr(self.manager, config.MGR_FC_GET_PRJ)(key)

    def set_project(self, key: str, project):
        return getattr(self.manager, config.MGR_FC_SYNC_PRJ)(key, project)

    def shutdown_server(self):
        return getattr(self.manager, config.MGR_FC_SHUTDOWN)()
