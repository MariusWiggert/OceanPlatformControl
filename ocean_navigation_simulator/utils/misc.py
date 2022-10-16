import contextlib
import os
import socket
import time
from types import SimpleNamespace
from typing import Optional

import psutil
import pynvml
import requests
from c3python import C3Python


@contextlib.contextmanager
def timing(string, verbose: Optional[int] = 0):
    """ Simple tool to check how long a specific code-part takes."""
    if verbose > 0:
        start = time.time()
        yield
        print(string.format(time.time()-start))
    else:
        yield

def get_c3(verbose: Optional[int] = 0):
    if not hasattr(get_c3, "c3"):
        with timing('Utils: Connect to c3 ({:.1f}s)', verbose):
            get_c3.c3 = C3Python(
                # Old Tag:
                # url='https://dev01-seaweed-control.c3dti.ai',
                # tag='dev01',
                # New tag:
                url='https://devseaweedrc1-seaweed-control.devrc01.c3aids.cloud',
                tenant='seaweed-control',
                tag='devseaweedrc1',
                keyfile='setup/keys/c3-rsa-jerome',
                username='jeanninj@berkeley.edu',
            ).get_c3()
    return get_c3.c3

def get_process_information_dict() -> dict:
    try:
        pynvml.nvmlInit()
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
    except Exception as e:
        gpu_info = SimpleNamespace(total=0, free=0, used=0)

    return {
        'process_pid': os.getpid(),
        'process_public_ip': requests.get('https://api.ipify.org').content.decode('utf8'),
        'process_private_ip': socket.gethostbyname(socket.gethostname()),
        'process_ram': f'{psutil.Process().memory_info().rss / 1e6:.1f}MB',
        'process_gpu_total': f'{gpu_info.total / 1e6:,.0f}MB',
        'process_gpu_used': f'{gpu_info.free / 1e6:,.0f}MB',
        'process_gpu_free': f'{gpu_info.used / 1e6:,.0f}MB',
    }

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
