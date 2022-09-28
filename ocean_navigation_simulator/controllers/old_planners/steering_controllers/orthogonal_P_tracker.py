import bisect
import math

import numpy as np

from ocean_navigation_simulator.steering_controllers.waypoint_track_contr import (
    WaypointTrackingController,
)


class orthogonal_P_tracker(WaypointTrackingController):
    """Simple proportional controller:
    Outside a radius r from the waypoint: actuate full power
    Inside the radius actuate: linearly decreasing actuation
    """

    def __init__(self, traj_data=None):
        super().__init__(traj_data=traj_data)
        self.waypoint_timings = None
        self.problem = None

    def set_waypoints(self, waypoints, problem=None):
        self.waypoints = waypoints
        if waypoints is not None:
            self.waypoint_timings = [point[2] for point in self.waypoints]
        else:
            self.waypoint_timings = None
        self.problem = problem

    def get_next_waypoint(self, state):
        """Get the next waypoint to actuate towards.
        Returns:
            lon, lat of waypoint that is ahead of the current time
        """
        idx = bisect.bisect_right(self.waypoint_timings, state[3])
        if idx >= len(self.waypoints):
            idx = -1
        return self.waypoints[idx][0], self.waypoints[idx][1]

    def get_next_action(self, state):
        """Returns (thrust, header) for the next timestep
        Args:
            state:
                A four element list describing the current state, i.e. [[lon],[lat], [battery_level], [time]]. Note each
                 nested variable is a float.
        Returns:
            An array containing the thrust and heading, i.e. array([thrust], [heading]).
        """

        # Step 1: get waypoint to actuate towards
        wy_pt_lon, wy_pt_lat = self.get_next_waypoint(state)

        # Step 2: estimate currents based on previous trajectories
        # lon, lat, battery time
        move_vec = np.zeros(2)
        u_last = np.zeros(2)
        if self.traj_data.trajectory.shape[1] > 1:
            move_vec = self.traj_data.trajectory[0:2, -1] - self.traj_data.trajectory[0:2, -2]
            # thrust, heading
            u_last = self.traj_data.control_traj[:, -1]

        # convert u_last to a lon, lat pair
        u_last_dir = np.array([np.cos(u_last[1]), np.sin(u_last[1])])

        # calculate the previous contributions from current by subtracting off the
        # projection of the movement vector onto the u direction
        curr_vec = move_vec  # (move_vec - np.dot(move_vec, u_last_dir) * u_last_dir)
        curr_norm = np.linalg.norm(curr_vec)
        if curr_norm > 0:
            curr_vec = curr_vec / np.linalg.norm(curr_vec)

        # move_dir = move_vec / np.linalg.norm(move_vec)
        # self.traj_data.trajectory[n, -1] - self.traj_data.trajectory[m, -2]

        # Step 3: actuate towards the next setpoint at full power
        dlon = wy_pt_lon - state[0][0]
        dlat = wy_pt_lat - state[1][0]
        mag = math.sqrt(dlon * dlon + dlat * dlat)

        # actuate towards waypoint full power
        u_vec = np.array([[dlon], [dlat]]) / mag

        # print("curr_vec shape", curr_vec.shape)
        # subtract off the projection of u onto the current direction
        u_dir = u_vec - np.dot(u_vec.reshape(1, 2), curr_vec.reshape(2, 1)) * curr_vec.reshape(2, 1)

        if np.linalg.norm(u_dir) > 1.0:
            u_dir /= np.linalg.norm(u_dir)
        # print("u_dir shape", u_dir.shape)
        # print("u_vec shape", u_vec.shape)

        u_out = super().transform_u_dir_to_u(u_dir=u_dir)
        return u_out

    def __str__(self):
        return "orthogonal_P_tracker"
