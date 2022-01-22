import numpy as np
import time
from jax.interpreters import xla
import jax.numpy as jnp
from jax import grad, jit, vmap
import psutil
# xla._xla_callable.cache_clear()

def print_ram():
    print("CPU usage percent: ", psutil.cpu_percent())
    # you can have the percentage of used RAM
    print("RAM usage percent: ", psutil.virtual_memory().percent)


def f(x):
    return jnp.mean(jnp.dot(x, x.T))

size = 1002
for i in range(1000):
    if i % 100 == 0: print(f"\nROUND #{i}")

    size -= 1
    a = jnp.ones([size, size])

    jit_f = None
    time.sleep(0.01)
    # xla._xla_callable.cache_clear()

    if i % 100 == 0: print("MEM:", print_ram())
    jit_f = jit(f)
    out = jit_f(a)
    if i % 100 == 0: print(out)
    if i % 100 == 0: print("MEM:", print_ram())