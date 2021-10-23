import matplotlib.pyplot as plt
import numpy as np


def plot_2D_traj(x_traj, title="Planned Trajectory"):
    plt.figure(1)
    plt.plot(x_traj[0, :], x_traj[1, :], '--', marker='o')
    plt.plot(x_traj[0, 0], x_traj[1, 0], '--', marker='x', color='red')
    plt.plot(x_traj[0, -1], x_traj[1, -1], '--', marker='x', color='green')
    plt.title(title)
    plt.xlabel('lon')
    plt.ylabel('lat')
    # plt.grid()
    plt.show()


def plot_opt_ctrl(times, ctrl_seq, title='Planned Optimal Control'):
    plt.figure(2)
    plt.step(times, ctrl_seq[0, :], where='post', label='u_power')
    plt.step(times, ctrl_seq[1, :], where='post', label='angle')
    plt.title(title)
    plt.ylabel('u_power and angle in units')
    plt.xlabel('time')
    # plt.grid()
    plt.show()


def plot_opt_results(T,u,x,N):
    plt.figure(3)
    plt.plot(np.linspace(0., T, N), u[0, :], '-.')
    plt.plot(np.linspace(0., T, N), u[1, :], '-.')
    plt.title('Results from ipopt Optimization')
    plt.xlabel('time in h')
    plt.ylabel('Actuation velocity in m/s')
    plt.legend(['u_x trajectory', 'u_y trajectory'])
    # plt.grid()
    plt.show()
    plot_2D_traj(x)
    return
