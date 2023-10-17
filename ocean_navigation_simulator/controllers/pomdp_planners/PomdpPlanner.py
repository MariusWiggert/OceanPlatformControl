import numpy as np

from ocean_navigation_simulator.controllers.pomdp_planners.DynamicsAndObservationModels import BaseDynObsModelParticles
from ocean_navigation_simulator.controllers.pomdp_planners.GenerativeParticleFilter import GenerativeParticleFilter
# from ocean_navigation_simulator.controllers.pomdp_planners.PFTDPWPlanner import PFTDPWPlanner
from ocean_navigation_simulator.controllers.pomdp_planners.PFTDPWPlanner_maxA import PFTDPWPlanner
from ocean_navigation_simulator.controllers.pomdp_planners.ParticleBelief import ParticleBelief
from ocean_navigation_simulator.controllers.pomdp_planners.ParticleFilterObserver import ParticleFilterObserver
from ocean_navigation_simulator.environment.Arena import ArenaObservation
from ocean_navigation_simulator.environment.Platform import PlatformAction


class PomdpPlanner:
    def __init__(self, x_target, particle_vel_func, stoch_params, F_max,
                 t_init, init_state, obs_noise,
                 key,
                 particle_filter_dict,
                 mcts_dict,
                 initial_weights=None):
        """
        :param x_target: jnp.array of center of target (x, y)
        :param particle_vel_func: function that gives velocity vel(particle_state)
        :param stoch_params: set of stochastic parameters that are used in vel_func (n_mc, n_params)
        :param F_max: maximum speed of the platform
        :param t_init: starting time or current time
        :param init_state: estimate of the current position at t (np.array (2,1))
        :param obs_noise: the assumed variance of the observation noise (assuming mean zero)
        :param key: random key to ensure reproducibility (for MCTS search)
        :param reward_func: function that gives the reward for a given state (x,y) and time t
        """
        self.vel_func = particle_vel_func
        self.x_target = x_target
        self.stoch_params = stoch_params
        self.F_max = F_max

        self.t_init = t_init
        self.x_est = init_state

        self.obs_noise = obs_noise

        self._key = key
        self.stoch_param_history = [stoch_params]

        self.particle_filter_dict = particle_filter_dict
        self.mcts_dict = mcts_dict

        # Create initial particles
        # all equally weighted initial particles with same init_state are
        init_state_tiled = np.tile(np.concatenate((init_state, np.array([t_init]))), (stoch_params.shape[0], 1))
        initial_particles = np.concatenate((init_state_tiled, stoch_params), axis=1)

        # Setting up necessary sub-variables and routines
        initial_particle_belief = ParticleBelief(initial_particles, weights=initial_weights)

        # Create the outside particle observer for assimilation
        filter_dyn_obs_model = BaseDynObsModelParticles(
            obs_noise=obs_noise, F_max=F_max, dt_dynamics=particle_filter_dict['dt_observations'],
            n_euler_per_dt_dynamics=particle_filter_dict.get('n_euler_per_dt_dynamics', 10),
            particle_vel_func=particle_vel_func, key=key,
            n_actions=mcts_dict['n_actions'], n_states=mcts_dict['n_states'])

        self.particle_observer = ParticleFilterObserver(
            particle_belief=initial_particle_belief,
            dynamics_and_observation=filter_dyn_obs_model,
            resample=particle_filter_dict['resample'],
            no_position_uncertainty=particle_filter_dict['no_position_uncertainty'])

        # initialize the MCTS planner
        mcts_dyn_obs_model = BaseDynObsModelParticles(
            obs_noise=obs_noise, F_max=F_max, dt_dynamics=mcts_dict['dt_mcts'],
            n_euler_per_dt_dynamics=mcts_dict.get('n_euler_per_dt_dynamics', 10),
            particle_vel_func=particle_vel_func, key=key,
            n_actions=mcts_dict['n_actions'], n_states=mcts_dict['n_states'])

        generative_particle_filter = GenerativeParticleFilter(
            dynamics_and_observation=mcts_dyn_obs_model,
            resample=mcts_dict['resample'],
            no_position_uncertainty=mcts_dict['no_position_uncertainty'])

        self.mcts_planner = PFTDPWPlanner(
            generative_particle_filter=generative_particle_filter,
            reward_function=mcts_dict['reward_function'],
            rollout_policy=mcts_dict.get('rollout_policy', None),
            rollout_value=mcts_dict.get('rollout_value', None),
            specific_settings=mcts_dict['mcts_settings'])

    def assimilate(self, x_obs, action) -> None:
        """
        :param x_obs: position observation (np.array (2,1))
        :param action: action taken from last state to get to this observation
        """
        # Update observer
        self.particle_observer.full_bayesian_update(action=action, observation=x_obs)

        # Add new params to history
        self.stoch_param_history.append(
            self.particle_observer.particle_belief_state.states[:, (self.mcts_dict['n_states'] + 1):])

    def get_action(self, observation: ArenaObservation) -> PlatformAction:
        """Given an observation, outputs the controller's next action
        Args:
          observation: observed state from simulator or other source (i.e. observer, other controller)
        Returns:
          Controller's next action as a PlatformAction object.
        """

        # Step 1: sample belief particles for planning
        planning_belief = self.particle_observer.get_planner_particle_belief(
            self.mcts_dict['num_planner_particles'])
        # run the MCTS planner for planning
        next_action = self.mcts_planner.get_best_action(planning_belief)

        return PlatformAction.from_discrete_action(next_action)