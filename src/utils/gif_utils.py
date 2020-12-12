import casadi as ca
from src.utils import particles, kernels
import parcels as p
import glob, imageio
from datetime import timedelta
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

def gif_straight_line(name, x_0, x_T, T, dt, fieldset, N_pictures=20, u_max=1., conv_m_to_deg=111120.):
    project_dir = '/Volumes/Data/2_Work/2_Graduate_Research/03_Tomlin/6_Seaweed/OceanPlatformControl'

    pset = p.ParticleSet.from_list(fieldset=fieldset,   # the fields on which the particles are advected
                                 pclass=particles.TargetParticle,  # the type of particles (JITParticle or ScipyParticle)
                                 lon=[x_0[0]],    # a vector of release longitudes
                                 lat=[x_0[1]],   # a vector of release latitudes
                                lon_target=[x_T[0]],
                                lat_target=[x_T[1]],
                                v_max=[u_max/conv_m_to_deg])

    # visualize as a gif
    straight_line_actuation = pset.Kernel(kernels.straight_line_actuation)
    for cnt in range(int(N_pictures)):
        # First plot the particles
        pset.show(savefile=project_dir + '/viz/pics_2_gif/particles'+str(cnt).zfill(2), field='vector', land=True, vmax=1.0, show_time=0.)

        pset.execute(p.AdvectionRK4 + straight_line_actuation,  # the kernel (which defines how particles move)
                     runtime=timedelta(hours=(T/3600.)/N_pictures),  # the total length of the run
                     dt=timedelta(seconds=dt),  # the timestep of the kernel
                    )

    file_list = glob.glob(project_dir + "/viz/pics_2_gif/*")
    file_list.sort()

    gif_file = project_dir + '/viz/gifs/' + name + '.gif'
    with imageio.get_writer(gif_file, mode='I') as writer:
        for filename in file_list:
            image = imageio.imread(filename)
            writer.append_data(image)
    print("saved gif as " + name)


def gif_open_loop_control(name, x_0, u, time_vec, T, dt, fieldset, N_pictures=20, u_max=1., conv_m_to_deg=111120.):
    # create the vehicle as a particle
    project_dir = '/Volumes/Data/2_Work/2_Graduate_Research/03_Tomlin/6_Seaweed/OceanPlatformControl'
    pset = p.ParticleSet.from_list(fieldset=fieldset,   # the fields on which the particles are advected
                             pclass=particles.OpenLoopParticle,  # the type of particles (JITParticle or ScipyParticle)
                             lon=[x_0[0]],    # a vector of release longitudes
                             lat=[x_0[1]],   # a vector of release latitudes
                            control_traj=[u],
                            control_time=[time_vec],
                            v_max=[u_max/conv_m_to_deg])
    open_loop_actuation = pset.Kernel(kernels.open_loop_control)

    for cnt in range(N_pictures):
        # First plot the particles
        pset.show(savefile=project_dir + '/viz/pics_2_gif/particles' + str(cnt).zfill(2), field='vector', land=True,
                  vmax=1.0, show_time=0.)

        pset.execute(p.AdvectionRK4 + open_loop_actuation,  # the kernel (which defines how particles move)
                     runtime=timedelta(hours=(T / 3600.) / N_pictures),  # the total length of the run
                     dt=timedelta(seconds=dt),  # the timestep of the kernel
                     )

    file_list = glob.glob(project_dir + "/viz/pics_2_gif/*")
    file_list.sort()

    gif_file = project_dir + '/viz/gifs/' + name + '.gif'
    with imageio.get_writer(gif_file, mode='I') as writer:
        for filename in file_list:
            image = imageio.imread(filename)
            writer.append_data(image)
    print("saved gif as " + name)