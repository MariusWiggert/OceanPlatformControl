import math

import matplotlib.pyplot as plt
import numpy as np

list_funcs = []

x = np.arange(-5, 5, 0.01)
# plt.plot(x, x ** 2)
# plt.plot(x, abs(2 * x))
list_funcs.append((lambda x: x ** 2, lambda x: 2 * x, "Mean squared error"))

# plt.figure()
# plt.plot(x, abs(x))
deriv_abs = lambda a: -1 if a < 0 else (1 if a > 0 else math.nan)
# plt.plot(x, list(map(deriv_abs, x)))
list_funcs.append((abs, deriv_abs, "Mean absolute error"))

plt.figure()
delta = 1.5
huber = lambda a, delta: 0.5 * abs(a) ** 2 if abs(a) < delta else delta * (abs(a) - 0.5 * delta)
deriv_huber = lambda a, delta: a if abs(a) < delta else delta * a / abs(a)
# plt.plot(x, [huber(a, delta) for a in x])
# plt.plot(x, [deriv_huber(a, delta) for a in x])
list_funcs.append((lambda x: huber(x, delta), lambda x: deriv_huber(x, delta), f"Huber(delta={delta})"))

for func, deriv, name in list_funcs:
    fig, ax = plt.subplots()
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    y = list(map(func, x)) + list(map(deriv, x))
    line1, = plt.plot(x, list(map(func, x)), color='red', label=f"{name}")
    line2, = plt.plot(x, list(map(deriv, x)), color='orange', label=f"derivative of {name}")
    ax.legend(handles=[line1, line2])
    plt.xticks(np.arange(min(x), max(x) + 1, 1))
    plt.yticks(np.arange(min(y), max(y) + 1, 1))
    ax.minorticks_on()
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.xlabel('x')
    plt.ylabel('f(x)')
