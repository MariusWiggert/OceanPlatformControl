import contextlib
import logging
import os
import socket
import time
from types import SimpleNamespace
from typing import Optional

import psutil
import pynvml
import requests
from c3python import C3Python

## How to get c3 Keyfile set up
# Step 1:   generate the public and private keys locally on your computer
#           in terminal run 'openssl genrsa -out c3-rsa.pem 2048' -> this generates the private key in the c3-rsa.pem file
#           for public key from it run 'openssl rsa -in c3-rsa.pem -outform PEM -pubout -out public.pem'
# Step 2:   move the c3-rsa.pem file to a specific folder
# Step 3:   Log into C3, start jupyter service and in a cell update your users public key by
#               usr = c3.User.get("mariuswiggert@berkeley.edu")
#               usr.publicKey = "<public key from file>"
#               usr.merge()


### Getting C3 Object for data downloading ###
def get_c3(verbose: Optional[int] = 0):
    """Helper function to get C3 object for access to the C3 Database"""

    KEYFILE = "setup/keys/c3-rsa-marius.pem"
    USERNAME = "mariuswiggert@berkeley.edu"

    if not hasattr(get_c3, "c3"):
        logging.getLogger("c3python.c3python").setLevel(logging.WARN)
        with timing("Utils: Connect to c3 ({:.1f}s)", verbose):
            get_c3.c3 = C3Python(
                # Old Tag: url='https://dev01-seaweed-control.c3dti.ai', tag='dev01',
                url="https://devseaweedrc1-seaweed-control.devrc01.c3aids.cloud",
                tag="devseaweedrc1",
                tenant="seaweed-control",
                keyfile=KEYFILE,
                username=USERNAME,
            ).get_c3()
    return get_c3.c3


@contextlib.contextmanager
def timing(string, verbose: Optional[int] = 1):
    """
    Simple tool to check how long a specific code-part takes.
    :arg
        string:
        verbose:
    """
    if verbose > 0:
        start = time.time()
        yield
        print(string.format(time.time() - start))
    else:
        yield


def get_process_information_dict() -> dict:
    """
    Helper function to get important process information as a dictionary.
    Runs without error when no GPU is installed.
    Returns:
        pid
        public ip
        private ip
        ram
        gpu total
        gpu used
        gpu free
    """
    try:
        pynvml.nvmlInit()
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
    except Exception:
        gpu_info = SimpleNamespace(total=0, free=0, used=0)

    return {
        "process_pid": os.getpid(),
        "process_public_ip": requests.get("https://api.ipify.org").content.decode("utf8"),
        "process_private_ip": socket.gethostbyname(socket.gethostname()),
        "process_ram": f"{psutil.Process().memory_info().rss / 1e6:.1f}MB",
        "process_gpu_total": f"{gpu_info.total / 1e6:,.0f}MB",
        "process_gpu_used": f"{gpu_info.free / 1e6:,.0f}MB",
        "process_gpu_free": f"{gpu_info.used / 1e6:,.0f}MB",
    }


class bcolors:
    """
    Helper class to use colors in the console.
    Example:
        print(f'{bcolors.FAIL} test {bcolors.ENDC}')
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @staticmethod
    def green(string: str) -> str:
        return f"{bcolors.OKGREEN}{string}{bcolors.ENDC}"

    @staticmethod
    def orange(string: str) -> str:
        return f"{bcolors.WARNING}{string}{bcolors.ENDC}"

    @staticmethod
    def red(string: str) -> str:
        return f"{bcolors.FAIL}{string}{bcolors.ENDC}"
