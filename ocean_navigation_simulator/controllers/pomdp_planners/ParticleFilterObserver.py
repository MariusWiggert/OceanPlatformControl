import numpy as np
from copy import deepcopy

from ocean_navigation_simulator.controllers.pomdp_planners.ParticleBelief import (
    ParticleBelief,
)
from ocean_navigation_simulator.controllers.pomdp_planners.ParticleFilterBase import (
    ParticleFilterBase,
)


class ParticleFilterObserver(ParticleFilterBase):
    def __init__(
            self,
            particle_belief: ParticleBelief,
            dynamics_and_observation,
            resample: bool,
            no_position_uncertainty: bool = False,
    ) -> None:
        super().__init__()

        # Keep track of particle belief state within the class as an observer
        self.particle_belief_state = particle_belief
        self.particle_belief_state.normalize_weights()

        # Dynamics, observation, resampling
        self.dynamics_and_observation = dynamics_and_observation
        self.resample = resample
        self.no_position_uncertainty = no_position_uncertainty

    def dynamics_update(
            self,
            actions: np.array
    ) -> None:
        # Dynamics-update the particle belief state
        self.particle_belief_state.update_states(
            self.dynamics_and_observation.get_next_states(
                states=self.particle_belief_state.states,
                actions=actions
            )
        )

    def observation_update(
            self,
            observations: np.array
    ) -> None:
        # Observation-update the particle belief state
        new_weight_multiplier = self.dynamics_and_observation.evaluate_observations(
            states=self.particle_belief_state.states,
            observations=observations
        )
        self.particle_belief_state.update_weights(
            self.particle_belief_state.weights * new_weight_multiplier
        )

    def full_bayesian_update(
            self,
            action: int,
            observation: np.array
    ) -> None:
        # Tile action and observation for vectorization
        actions = np.tile(action, self.particle_belief_state.num_particles)
        observation_flattened = observation.flatten()
        observations = np.tile(observation_flattened, [self.particle_belief_state.num_particles, 1])

        # Run dynamics and observation updates
        self.dynamics_update(actions)
        self.observation_update(observations)

        # implement position certainty
        if self.no_position_uncertainty:
            self.particle_belief_state.states[:, :2] = observations
            # self.particle_belief_state.states = np.concatenate(
            #     (observations, self.particle_belief_state.states[:, 2:]), axis=1)

        # Normalize and resample
        self.particle_belief_state.normalize_weights()
        if self.resample:
            self._resample_particles()

    def _resample_particles(self) -> None:
        # Naive resampling: essentially just re-sampling from the current belief states.
        # This results in eliminating low weight particles and duplicating high weight particles.
        self.particle_belief_state.normalize_weights()
        sampled_indices = np.random.choice(
            self.particle_belief_state.num_particles,
            self.particle_belief_state.num_particles,
            p=self.particle_belief_state.weights
        )
        self.particle_belief_state.update_states(
            self.particle_belief_state.states[sampled_indices]
        )
        self.particle_belief_state.reinitialize_weights()

    def get_planner_particle_belief(
            self,
            num_particles_planning: int
    ) -> ParticleBelief:
        # Return a simple copy if number of particles for planning is same
        if num_particles_planning == self.particle_belief_state.num_particles:
            return deepcopy(self.particle_belief_state)

        # Return a down sampled copy for planning
        self.particle_belief_state.normalize_weights()
        sampled_indices = np.random.choice(
            self.particle_belief_state.num_particles,
            num_particles_planning,
            p=self.particle_belief_state.weights
        )

        return ParticleBelief(
            self.particle_belief_state.states[sampled_indices].copy(),
            np.ones(num_particles_planning)
        )