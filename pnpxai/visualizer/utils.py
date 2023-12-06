from subprocess import Popen
import os
import pathlib
import sys
from typing import List


def start_detached_process(cmd: List[str]):
    config = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} \
        if os.name == 'nt' else \
        {"preexec_fn": os.setpgrp}

    Popen(cmd, **config)
