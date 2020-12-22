import matplotlib.pyplot as plt
import numpy as np


def plot_opt_results(T,u,x,N):
    plt.figure(1)
    plt.plot(np.linspace(0., T, N), u[0, :], '-.')
    plt.plot(np.linspace(0., T, N), u[1, :], '-.')
    plt.title('Results from ipopt Optimization')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend(['u_x trajectory', 'u_y trajectory'])
    # plt.grid()
    plt.show()

    plt.figure(1)
    plt.plot(x[0, :], x[1, :], '--')
    plt.title('Trajectory ipopt')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.grid()
    plt.show()
    return