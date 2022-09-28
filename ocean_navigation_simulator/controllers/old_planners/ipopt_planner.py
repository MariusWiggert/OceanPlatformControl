import bisect

import casadi as ca
import numpy as np

from ocean_navigation_simulator.planners.planner import Planner
from ocean_navigation_simulator.utils import plotting_utils, simulation_utils


class IpoptPlanner(Planner):
    """Planner based on non-linear optimization in ipopt

    Attributes required in the specific_settings dict
        t_init_in_h:
            The initial time to be used in the optimization (only fixed time Opt. implemented right now)
        n_dec_var:
            The number of decision variables in the discrete optimization problem.
        temporal_stride:
            Integer. If the forecasts are hourly but because bspline fitting is expensive we only
            want to take every 5h currents in fitting it. Essentially saves compute speed.

        see Planner class for the rest of the attributes.
    """

    def __init__(self, problem, gen_settings, specific_settings):
        # initialize superclass
        super().__init__(problem, gen_settings, specific_settings)

        # initialize values
        self.T_goal_in_h = self.specific_settings["T_goal_in_h"] * 3600
        self.N = self.specific_settings["n_dec_var"]

    def plan(self, x_t, trajectory=None):

        # Step 1: read the relevant subset of data
        if self.new_forecast_dicts:
            print("ipopt Planner: New forecast file so reloading data.")
            t_interval, lat_bnds, lon_bnds = simulation_utils.convert_to_lat_lon_time_bounds(
                x_t,
                x_T,
                deg_around_x0_xT_box=self.specific_settings["deg_around_xt_xT_box"],
            )
            self.grids_dict, u_data, v_data = simulation_utils.get_current_data_subset(
                self.cur_forecast_dicts, t_interval, lat_bnds, lon_bnds
            )

            # Step 2: get the current interpolation functions
            self.u_curr_func, self.v_curr_func = simulation_utils.get_interpolation_func(
                self.grids_dict, u_data, v_data, self.gen_settings["int_pol_type"], self.fixed_time
            )

            self.new_forecast_file = False

        # Step 2: run optimization
        self.T, self.u_open_loop, self.x_solver, self.dt = self.run_optimization(x_t=x_t)
        # create the vector containing the times when which control is active
        self.control_time_vec = x_t[3] + np.arange(self.u_open_loop.shape[1] + 1) * self.dt

    def get_next_action(self, state):
        """When run in open loop this action is directly applied."""
        # an easy way of finding for each time, which index of control signal to apply
        idx = bisect.bisect_right(self.control_time_vec, state[3]) - 1
        if idx == len(self.control_time_vec) - 1:
            idx = idx - 1
            print("WARNING: continuing using last control although not planned as such")

        u_dir = np.array([[self.u_open_loop[0, idx]], [self.u_open_loop[1, idx]]])

        # transform to thrust & angle
        u_out = super().transform_u_dir_to_u(u_dir=u_dir)
        return u_out

    def run_optimization(self, x_t):
        """Set-up the ipopt problem and solve it.

        Input:
            x_t     the starting point of the optimization

        Returns:
            - T     total time from start to end
            - u     array of controls over time (u_x, u_y)
            - x     state vector lat,lon over time
            - dt    time between to points/control times
        """
        # implemented by the child classes
        pass

    def plot_opt_results(self):
        plotting_utils.plot_opt_results(
            self.T / 3600, self.u_open_loop * self.dyn_dict["u_max"], self.x_solver, self.N
        )

    def get_waypoints(self):
        """Returns: a list of waypoints each containing [lon, lat, time]"""
        return np.vstack((self.x_solver, self.control_time_vec)).T.tolist()


class IpoptPlannerVarCur(IpoptPlanner):
    """Non_linear optimizer using ipopt for current fields that are time-varying."""

    def __init__(self, problem, gen_settings, specific_settings):
        # initialize superclass
        super().__init__(problem, gen_settings, specific_settings)

    def run_optimization(self, x_t):
        # create optimization problem
        opti = ca.Opti()

        # declare fixed End time and variable t for indexing to variable current
        T = self.T_goal_in_h
        t = ca.MX.sym("t")  # time symbolic variable

        # declare decision variables
        # For now 2 State system (X, Y), not capturing battery or seaweed mass
        x = opti.variable(2, self.N + 1)  # Decision variables for state trajetcory
        u = opti.variable(2, self.N)

        # Parameters (not optimized over)
        x_start = opti.parameter(2, 1)
        x_goal = opti.parameter(2, 1)

        # optimizer objective
        opti.minimize((ca.dot(u[0, :], u[0, :]) + ca.dot(u[1, :], u[1, :])))

        # create the dynamics function (note here in form of u_x and u_y)
        F = ca.Function(
            "f",
            [t, x, u],
            [
                ca.vertcat(
                    u[0] * self.dyn_dict["u_max"] + self.u_curr_func(ca.vertcat(t, x[1], x[0])),
                    u[1] * self.dyn_dict["u_max"] + self.v_curr_func(ca.vertcat(t, x[1], x[0])),
                )
                / self.gen_settings["conv_m_to_deg"]
            ],
            # ,c_recharge - u[0]**2)],
            ["t", "x", "u"],
            ["x_dot"],
        )

        # add the dynamics constraints
        dt = T / self.N
        for k in range(self.N):
            # calculate time in POSIX seconds
            time = x_t[3] + k * dt
            # explicit forward euler version
            x_next = x[:, k] + dt * F(t=time, x=x[:, k], u=u[:, k])["x_dot"]
            opti.subject_to(x[:, k + 1] == x_next)

        # start state & goal constraint
        opti.subject_to(x[:, 0] == x_start)
        opti.subject_to(ca.dot(x[:2, -1] - x_goal, x[:2, -1] - x_goal) <= 0.001)

        # control constraints
        opti.subject_to(u[0, :] ** 2 + u[1, :] ** 2 <= 1)

        # State constraints to the map boundaries
        opti.subject_to(
            opti.bounded(self.grids_dict["x_grid"].min(), x[0, :], self.grids_dict["x_grid"].max())
        )
        opti.subject_to(
            opti.bounded(self.grids_dict["y_grid"].min(), x[1, :], self.grids_dict["y_grid"].max())
        )

        # battery constraint
        # opti.subject_to(opti.bounded(0.1, x[2, :], 1.))

        # Set the values for the optimization problem
        opti.set_value(x_start, x_t[:2])
        opti.set_value(x_goal, self.x_T)

        # Optional: initialize variables for the optimizer
        # opti.set_initial(x, x_init)
        # opti.set_initial(u, u_init)

        opti.solver("ipopt")
        sol = opti.solve()

        # extract the time vector and control signal
        T = sol.value(T)
        u = sol.value(u)
        x = sol.value(x)
        dt = sol.value(dt)

        return T, u, x, dt


class IpoptPlannerFixCur(IpoptPlanner):
    """Non_linear optimizer using ipopt for current fields that are fixed."""

    def __init__(self, problem, gen_settings, specific_settings):
        # initialize superclass
        super().__init__(problem, gen_settings, specific_settings)

    def run_optimization(self, x_t):
        # create optimization problem
        opti = ca.Opti()

        # fixed end_time
        T = self.T_goal_in_h

        # declare decision variables
        # For now 2 State system (X, Y), not capturing battery or seaweed mass
        x = opti.variable(2, self.N + 1)  # Decision variables for state trajetcory
        u = opti.variable(2, self.N)

        # Parameters (not optimized over)
        x_start = opti.parameter(2, 1)
        x_goal = opti.parameter(2, 1)

        # optimizer objective (minimize actuation effort)
        opti.minimize((ca.dot(u[0, :], u[0, :]) + ca.dot(u[1, :], u[1, :])))

        # create the dynamics function
        # Note: using u_x and u_y instead of thrust & heading for smoother optimization landscape
        F = ca.Function(
            "f",
            [x, u],
            [
                ca.vertcat(
                    u[0] * self.dyn_dict["u_max"] + self.u_curr_func(ca.vertcat(x[1], x[0])),
                    u[1] * self.dyn_dict["u_max"] + self.v_curr_func(ca.vertcat(x[1], x[0])),
                )
                / self.gen_settings["conv_m_to_deg"]
            ],
            # ,c_recharge - u[0]**2)],
            ["x", "u"],
            ["x_dot"],
        )

        # add the dynamics constraints
        dt = T / self.N
        for k in range(self.N):
            # explicit forward euler version
            x_next = x[:, k] + dt * F(x=x[:, k], u=u[:, k])["x_dot"]
            opti.subject_to(x[:, k + 1] == x_next)

        # Start_state & goal constraint
        opti.subject_to(x[:, 0] == x_start)
        opti.subject_to(ca.dot(x[:2, -1] - x_goal, x[:2, -1] - x_goal) <= 0.001)

        # control magnitude constraints
        opti.subject_to(u[0, :] ** 2 + u[1, :] ** 2 <= 1)

        # State constraints to the map boundaries
        opti.subject_to(
            opti.bounded(self.grids_dict["x_grid"].min(), x[0, :], self.grids_dict["x_grid"].max())
        )
        opti.subject_to(
            opti.bounded(self.grids_dict["y_grid"].min(), x[1, :], self.grids_dict["y_grid"].max())
        )

        # battery constraint
        # opti.subject_to(opti.bounded(0.1, x[2, :], 1.))

        # Set the values for the optimization problem
        opti.set_value(x_start, x_t[:2])
        opti.set_value(x_goal, self.x_T)

        # Optional: initialize variables for the optimizer
        # opti.set_initial(x, x_init)
        # opti.set_initial(u, u_init)

        opti.solver("ipopt")
        sol = opti.solve()

        # extract the time vector and control signal
        T = sol.value(T)
        u = sol.value(u)
        x = sol.value(x)
        dt = sol.value(dt)

        return T, u, x, dt
