import functools
import math
from typing import Tuple

import jax
import numpy as np
from matplotlib import pyplot as plt

from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.controllers.pomdp_planners.HJRolloutValue.dynamics import Platform2Dcurrents, \
    Platform2DcurrentsDiscrete
import jax.numpy as jnp
import hj_reachability as hj
from ocean_navigation_simulator.controllers.pomdp_planners.ParticleBelief import ParticleBelief


def anlt_dg_vel(t, x, y, A, eps, omega):
    a = eps * jnp.sin(omega * t)
    b = 1 - 2 * a
    f = a * (x ** 2) + b * x
    df = 2 * (a * x) + b
    return jnp.array([-jnp.pi * A * jnp.sin(jnp.pi * f) * jnp.cos(jnp.pi * y),
                      jnp.pi * A * jnp.cos(jnp.pi * f) * jnp.sin(jnp.pi * y) * df])


class ParticleHJValueEstimator:
    """Takes in stochastic particles and computes the HJ value function for each of them.
    Note: Quite specific implemented for now as extra class, we make it more general later on.

    Example specific_settings:
    specific_settings = {
        "n_time_vector": 200,
        "T_goal_in_units": 20,
        "discrete_actions": True,
        "grid_res": 0.05,
    }
    """

    def __init__(self, problem: NavigationProblem, specific_settings: dict, vel_func: callable):
        # save the variables
        self.times = None
        self.all_val_funcs = None
        self.problem = problem
        self.specific_settings = specific_settings
        self.vel_func = vel_func

        # set up everything so that once we get stochastic params, we can compute the value functions
        def dirichlet(x, pad_width: int):
            return jnp.pad(x, ((pad_width, pad_width)), "constant", constant_values=3.0)

        # compute grid resolution from spatial resolution
        x_res = int(
            (problem.x_range[1] - problem.x_range[0]) / self.specific_settings[
                'grid_res'])
        y_res = int(
            (problem.y_range[1] - problem.y_range[0]) / self.specific_settings[
                'grid_res'])

        # instantiate grid
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(lo=np.array([problem.x_range[0], problem.y_range[0]]),
                        hi=np.array([problem.x_range[1], problem.y_range[1]])),
            (x_res, y_res),
            boundary_conditions=(dirichlet, dirichlet))

        signed_distance = hj.shapes.shape_sphere(grid=self.grid,
                                                 center=self.problem.end_region.__array__(),
                                                 radius=self.problem.target_radius)
        self.term_val_func = np.maximum(signed_distance, np.zeros(signed_distance.shape))

        def multi_reach_step(mask, val):
            val = jnp.where(mask <= 0, -1, val)
            return val

        p_multi_reach_step = functools.partial(multi_reach_step, self.term_val_func)
        self.solver_settings = hj.SolverSettings.with_accuracy("high",
                                                               artificial_dissipation_scheme=
                                                               hj.artificial_dissipation.local_local_lax_friedrichs,
                                                               hamiltonian_postprocessor=p_multi_reach_step
                                                               )
        if self.specific_settings.get('discrete_actions', False):
            dynamics_class = Platform2DcurrentsDiscrete
        else:
            dynamics_class = Platform2Dcurrents

        def compute_val_func(times, stoch_param):
            def x_cur(arg, vel_params):
                return self.vel_func(arg[-1], *(arg[:-1]), *vel_params)[0]

            def y_cur(arg, vel_params):
                return self.vel_func(arg[-1], *(arg[:-1]), *vel_params)[1]

            x_cur = jax.jit(functools.partial(x_cur, vel_params=stoch_param))
            y_cur = jax.jit(functools.partial(y_cur, vel_params=stoch_param))

            self.dynamics_model = dynamics_class(u_max=self.problem.platform_dict["u_max_in_mps"],
                                                 control_mode='min',
                                                 x_current=x_cur,
                                                 y_current=y_cur)

            _, val_func = hj.solve(self.solver_settings, self.dynamics_model, self.grid, times,
                                   self.term_val_func, progress_bar=False)
            return val_func

        # prepare the vmap over multiple stochastic params
        self.compute_val_func = compute_val_func
        self.compute_val_funcs = jax.vmap(compute_val_func, (None, 0), 0)
        # this will keep times fixed, then map over 0th axis of stoch params and write in the 0th axis output

    def compute_hj_value_funcs(self, current_time, stoch_param):
        """Compute the HJ value functions for the given stochastic params and the given current time."""
        self.times = jnp.linspace(current_time + self.specific_settings['T_goal_in_units'],
                                  current_time,
                                  self.specific_settings['n_time_vector'])

        self.all_val_funcs = self.compute_val_funcs(self.times, stoch_param)

    def __call__(self, particleBelief: ParticleBelief):
        """Estimate the value of the rollout at the given particleBelief.

        :param particleBelief: particleBelief with states and weights
        :return: estimated value of the rollout (time-to-reach).
        """
        # Note: this assumes that all_val_funcs is ordered the same way as the particleBelief states
        hj_values = jax.vmap(self.grid.spacetime_interpolate,
                             in_axes=(None, 0, 0, 0),
                             out_axes=0)(self.times, self.all_val_funcs,
                                         particleBelief.states[:, 2],
                                         particleBelief.states[:, :2])

        # transform to time-to-reach
        ttr_values = self.specific_settings['T_goal_in_units'] + hj_values - particleBelief.states[:, 2]

        # now compute the reward values based on the hj_values
        reward_values = self.specific_settings['ttr_to_rewards'](ttr_values)

        # calculate the sum (because weights add to 1 already)
        return (reward_values * particleBelief.weights).sum()

    def plot_ttr_snapshot(
            self,
            stoch_params_idx: int,
            time_idx: int = -1,
            ax: plt.Axes = None,
            return_ax: bool = False,
            fig_size_inches: Tuple[int, int] = (12, 12),
            alpha_color: float = 1.0,
            time_to_reach: bool = True,
            granularity: float = 0.1,
            **kwargs,
    ):
        """Plot the reachable set the planner was computing last at  a specific rel_time_in_seconds.
        Args:
            stoch_params_idx:       the index of the stochastic parameter set to plot
            time_idx:               the time index of the value function to plot
            ax:                     Optional: axis object to plot on top of
            return_ax:              if true, function returns ax object for more plotting
            fig_size_inches:        Figure size
            ### Rest only relevant for multi-time-reach-back
            alpha_color:            the alpha level of the colors when plotting multi-time-reachability
            time_to_reach:          if True we plot the time-to-reach the target, otherwise the value function
            granularity:       the granularity of the color-coding
        """
        if self.grid.ndim != 2:
            raise ValueError("plot_reachability is currently only implemented for 2D sets")

        # create the axis object if not fed in
        if ax is None:
            ax = plt.axes()

        # get_initial_value
        initial_values = self.all_val_funcs[stoch_params_idx, 0, ...]

        # interpolate the value function to the specific time
        val_at_t = self.all_val_funcs[stoch_params_idx, time_idx, ...]

        if ("val_func_levels" not in kwargs):
            # compute the levels
            abs_time = self.times[0] - self.times[time_idx]
            n_levels = abs(math.ceil(abs_time / granularity)) + 1
            vmin = val_at_t.min()

            if vmin == 0 or n_levels == 1:
                val_func_levels = np.array([0, 1e-10])
                abs_time_y_ticks = np.array([0.0, 0.0])
            else:
                val_func_levels = np.linspace(vmin, 0, n_levels)
                abs_time_y_ticks = np.around(np.linspace(abs_time, 0, n_levels), decimals=1)

            if time_to_reach:
                y_label = "Fastest Time-to-Target in time-units"
                abs_time_y_ticks = np.abs(np.flip(abs_time_y_ticks, axis=0))
            else:
                y_label = "HJ Value Function"

            # return val_func_levels, abs_time_y_ticks, y_label

            # package them in kwargs
            kwargs.update(
                {
                    "val_func_levels": val_func_levels,
                    "y_label": y_label,
                    "yticklabels": abs_time_y_ticks,
                }
            )

        # plot the set on top of ax
        ax = hj.viz._visSet2D(
            self.grid,
            val_at_t,
            plot_level=0,
            color_level="black",
            colorbar=True,
            obstacles=None,
            target_set=initial_values,
            return_ax=True,
            input_ax=ax,
            alpha_colorbar=alpha_color,
            **kwargs,
        )

        ax.scatter(
            self.problem.start_state.lon.deg,
            self.problem.start_state.lat.deg,
            color="r",
            marker="o",
            zorder=6,
        )
        ax.scatter(
            self.problem.end_region.lon.deg,
            self.problem.end_region.lat.deg,
            color="g",
            marker="x",
            zorder=6,
        )

        ax.set_title(
            "Value Function at time {} units".format(
                self.times[time_idx]
            )
        )
        ax.set_facecolor("white")

        # adjust the fig_size
        fig = plt.gcf()
        fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
        if return_ax:
            return ax
        else:
            plt.show()
