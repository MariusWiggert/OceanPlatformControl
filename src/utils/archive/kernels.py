import math
import numpy as np


# Define a custom kernel to execute the trajectory
def straight_line_actuation(particle, fieldset, time):
    # find direction
    dlon = particle.lon_target - particle.lon
    dlat = particle.lat_target - particle.lat
    # angle = math.tan(dlat/dlon) # breaks sometimes when dlon=0

    mag = math.sqrt(dlon * dlon + dlat * dlat)

    # go there full power
    particle.lon += dlon/mag * particle.v_max * particle.dt
    particle.lat += dlat/mag * particle.v_max * particle.dt


def open_loop_control(particle, fieldset, time):
    # could be great to implement feedback policies

    # find index of the relevant control signal
    # index = np.where(time - particle.control_time < 0)[0].argmax()
    index = 1
    while particle.control_time[-index] - time > 0:
        index += 1

    particle.lon += particle.control_traj[0, -index + 1] * particle.v_max * particle.dt
    particle.lat += particle.control_traj[1, -index + 1] * particle.v_max * particle.dt

    # particle.lon += math.cos(particle.control_traj[1, -index + 1]) * particle.control_traj[
    #     0, -index + 1] * particle.v_max * particle.dt
    # particle.lat += math.sin(particle.control_traj[1, -index + 1]) * particle.control_traj[
    #     0, -index + 1] * particle.v_max * particle.dt
