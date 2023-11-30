import os
import sys
from subprocess import Popen
import subprocess
import pathlib

dir = pathlib.Path(__file__).parent.resolve()
cmd = [sys.executable, f"{dir}/test_server.py"]


config = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} \
    if os.name == 'nt' else \
    {"preexec_fn": os.setpgrp}

Popen(cmd,
      #  stdout=open('/dev/null', 'w'),
      #  stderr=open('logfile.log', 'a'),
      **config
      )

print('client exited')
exit()
