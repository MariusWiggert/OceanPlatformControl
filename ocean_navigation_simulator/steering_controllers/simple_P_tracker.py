from ocean_navigation_simulator.steering_controllers.waypoint_track_contr import WaypointTrackingController
import numpy as np
import math, bisect


class simple_P_tracker(WaypointTrackingController):
    """ Simple proportional controller:
        Actuate full power toward the next waypoint.
    """

    def __init__(self, traj_data=None):
        super().__init__(traj_data=traj_data)
        self.waypoint_timings = None

    def set_waypoints(self, waypoints, problem=None):
        self.waypoints = waypoints
        if waypoints is not None:
            self.waypoint_timings = [point[2] for point in self.waypoints]
        else:
            self.waypoint_timings = None

    def get_next_waypoint(self, state):
        """ Get the next waypoint to actuate towards.
        Returns:
            lon, lat of waypoint that is ahead of the current time
        """
        idx = bisect.bisect_right(self.waypoint_timings, state[3], hi=len(self.waypoints)-1)
        return self.waypoints[idx][0], self.waypoints[idx][1]

    def get_next_action(self, state):
        """ Returns (thrust, header) for the next timestep
        Args:
            state:
                A four element list describing the current state, i.e. [[lon],[lat], [battery_level], [time]]. Note each
                 nested variable is a float.
        Returns:
            An array containing the thrust and heading, i.e. array([thrust], [heading]).
        """

        # Step 1: get waypoint to actuate towards
        wy_pt_lon, wy_pt_lat = self.get_next_waypoint(state)

        # Step 2: actuate towards it full power
        dlon = wy_pt_lon - state[0][0]
        dlat = wy_pt_lat - state[1][0]
        mag = math.sqrt(dlon * dlon + dlat * dlat)

        # actuate towards waypoint full power
        u_dir = np.array([[dlon / mag], [dlat / mag]])
        u_out = super().transform_u_dir_to_u(u_dir=u_dir)
        return u_out

    def __str__(self):
        return "Simple P Controller"
