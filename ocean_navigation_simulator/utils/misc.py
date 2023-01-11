import contextlib
import itertools
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

c3_logger = logging.getLogger("c3")


### Getting C3 Object for data downloading ###
def get_c3():
    """Helper function to get C3 object for access to the C3 Database"""
    KEYFILE = "setup/keys/c3-rsa-marius.pem"
    USERNAME = "mariuswiggert@berkeley.edu"

    # reload after 10min to prevent c3 timeout
    if not hasattr(get_c3, "c3") or (
        not hasattr(get_c3, "timestamp") and (time.time() - get_c3.timestamp) > 600
    ):
        logging.getLogger("c3python.c3python").setLevel(logging.WARN)
        print("Starting to connect to c3")
        with timing_logger("Utils: Connect to c3 ({})", c3_logger, logging.INFO):
            get_c3.c3 = C3Python(
                # Old Tag: url='https://dev01-seaweed-control.c3dti.ai', tag='dev01',
                url="https://devseaweedrc1-seaweed-control.devrc01.c3aids.cloud",
                tag="devseaweedrc1",
                tenant="seaweed-control",
                keyfile=KEYFILE,
                username=USERNAME,
            ).get_c3()
            get_c3.timestamp = time.time()
    return get_c3.c3


@contextlib.contextmanager
def timing(string, verbose: Optional[int] = 1):
    """Simple tool to check how long a specific code-part takes."""
    start = time.time()
    yield
    exec_time = time.time() - start
    if exec_time > 1:
        text = f"{exec_time:.2f}s"
    else:
        text = f"{1000*exec_time:.2f}ms"
    if verbose > 0:
        print(string.format(text))


@contextlib.contextmanager
def timing_logger(string, logger: logging.Logger, level=logging.INFO):
    """
    Simple tool to check how long a specific code-part takes.
    :arg
        string:
        verbose:
    """
    start = time.time()
    yield
    exec_time = time.time() - start
    if exec_time > 1:
        text = f"{exec_time:.2f}s"
    else:
        text = f"{1000*exec_time:.2f}ms"
    logger.log(level, string.format(text))


@contextlib.contextmanager
def timing_dict(dict, field, string=None, logger: logging.Logger = None, level=logging.INFO):
    """
    Simple tool to check how long a specific code-part takes.
    :arg
        string:
        verbose:
    """
    start = time.time()
    yield
    exec_time = time.time() - start
    dict[field] += exec_time
    if None not in [string, logger]:
        if exec_time > 1:
            text = f"{exec_time:.2f}s"
        else:
            text = f"{1000*exec_time:.2f}ms"
        logger.log(level, string.format(text))


def set_arena_loggers(level):
    """helper function to set all relevant logger levels"""
    logging.getLogger("c3").setLevel(level)
    logging.getLogger("arena").setLevel(level)
    logging.getLogger("arena.factory").setLevel(level)
    logging.getLogger("arena.platform").setLevel(level)
    logging.getLogger("arena.controller").setLevel(level)
    logging.getLogger("observer").setLevel(level)

    logging.getLogger("data_source").setLevel(level)

    logging.getLogger("arena.ocean_field").setLevel(level)
    logging.getLogger("arena.ocean_field.ocean_source").setLevel(level)

    logging.getLogger("arena.ocean_field.seaweed_growth_source").setLevel(level)
    logging.getLogger("arena.seaweed_growth_field").setLevel(level)

    logging.getLogger("arena.solar_field").setLevel(level)
    logging.getLogger("arena.solar_field.analytical_source").setLevel(level)

    logging.getLogger("OceanEnv").setLevel(level)
    logging.getLogger("MissionGenerator").setLevel(level)


def silence_ray_and_tf():
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.FATAL)
    logging.getLogger("absl").setLevel(logging.FATAL)
    logging.getLogger("external").setLevel(logging.FATAL)
    logging.getLogger(
        "external/org_tensorflow/tensorflow/compiler/xla/stream_executor/cuda"
    ).setLevel(logging.FATAL)
    import warnings

    warnings.simplefilter("ignore", UserWarning)

    logging.getLogger("ray").setLevel(logging.WARN)
    logging.getLogger("rllib").setLevel(logging.WARN)
    logging.getLogger("policy").setLevel(logging.WARN)


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


def get_markers():
    return itertools.cycle((".", "+", "*", "x", "p", "h", "d", "1", "P", "H"))


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
        return f"{bcolors.FAIL}{str(string)}{bcolors.ENDC}"
