from multiprocessing.managers import BaseManager
from subprocess import Popen
import time
import os
import pathlib
import sys


class Client(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Client, cls).__new__(cls)
        return cls.instance

    def __init__(
        self,
        address: str = '',
        port: int = 37844,
        password: bytearray = b"cderfv34"
    ):
        self.manager = None
        self.start_or_connect_to_server(
            address=address,
            port=port,
            password=password
        )

    def get_init_manager(self, address: str, port: int, password: bytearray):
        return BaseManager((address, port), password)

    def connect_to_server(self):
        try:
            self.manager.connect()
            print("Connected to the server")
        except ConnectionRefusedError as e:
            return False
        return True

    def start_server(self):
        dir = pathlib.Path(__file__).parent.resolve()
        cmd = [sys.executable, f"{dir}/test_server.py"]

        config = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} \
            if os.name == 'nt' else \
            {"preexec_fn": os.setpgrp}

        Popen(cmd, **config)

    def create_server_process(self, address: str, port: int, password: bytearray):
        pid = os.fork()
        if pid:
            # parent (client) process
            while not self.manager.connect():
                time.sleep(0.1)
        else:
            # child (server) process
            self.start_server()

    def start_or_connect_to_server(self, address: str, port: int, password: bytearray):
        self.manager = self.get_init_manager(address, port, password)
        try:
            self.manager.connect()
        except ConnectionRefusedError as e:
            self.create_server_process(address, port, password)
