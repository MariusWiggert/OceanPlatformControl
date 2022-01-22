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




# @profile
# def main():
#     @jit
#     def f(x):
#         return x
#
#     def ff(x):
#         return f(x)
#     # Main function
#     ff_jit = jit(ff)
#
#     a = ff_jit(jnp.ones((7)))
#     b = ff_jit(5)
#     c = ff_jit(jnp.ones((2,2)))
#     print("ff_jit._cache_size()", ff_jit._cache_size())
#     # ff_jit._clear_cache()
#     del ff_jit
#     del f
#     del ff
#     del a, b, c


@profile
def main():
    from jax import numpy as jnp
    from jax import random
    from jax.interpreters import xla
    key = random.PRNGKey(42)
    shape = (int(2e+3), int(1e+2))
    volume_weight = jnp.linspace(1., 1e+2, num=shape[1])
    line_of_sight_weight = jnp.exp(random.normal(key, shape))

    @jit
    def loss_bwd(_, t):
        xi_w_vol = t * line_of_sight_weight.T
        wo_cumsum = jnp.flip(jnp.cumsum(jnp.flip(xi_w_vol, 0), axis=0), 0)
        xi = volume_weight[:, jnp.newaxis] * wo_cumsum
        return (xi, )

    # jrev = jit(loss_bwd)
    # a = jrev(None, 1)
    # b = jrev(None, 1.)
    # print(jrev._cache_size())
    # jrev._clear_cache()
    # del a, b
    # del jrev
    # xla._xla_callable.cache_clear()
    a = loss_bwd(None, 1)
    b = loss_bwd(None, 1.)
    print(loss_bwd._cache_size())
    loss_bwd._clear_cache()
    del a, b
    del loss_bwd
    xla._xla_callable.cache_clear()


if __name__ == '__main__':
    main()

