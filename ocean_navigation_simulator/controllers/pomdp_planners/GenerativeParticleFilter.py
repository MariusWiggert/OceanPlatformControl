import numpy as np
from copy import deepcopy

from ocean_navigation_simulator.controllers.pomdp_planners.ParticleBelief import (
    ParticleBelief,
)
from ocean_navigation_simulator.controllers.pomdp_planners.ParticleFilterBase import (
    ParticleFilterBase,
)

class GenerativeParticleFilter(ParticleFilterBase):
    def __init__(
        self, 
        dynamics_and_observation, 
        resample: bool
    ) -> None:
        super().__init__()

        # Dynamics, observation, resampling
        self.dynamics_and_observation = dynamics_and_observation
        self.resample = resample

    def dynamics_update(
        self, 
        particle_belief_state: ParticleBelief,
        actions: np.array
    ) -> ParticleBelief:
        # Dynamics-update the input particle belief state
        particle_belief_state.update_states(
            self.dynamics_and_observation.get_next_states(
                particle_belief_state.states,
                actions
            )
        )
        return particle_belief_state

    def observation_update(
        self, 
        particle_belief_state: ParticleBelief,
        observations: np.array
    ) -> ParticleBelief:
        # Observation-update the input particle belief state
        new_weight_multiplier = self.dynamics_and_observation.evaluate_observations(
            particle_belief_state.states,
            observations    
        )
        particle_belief_state.update_weights(
            particle_belief_state.weights * new_weight_multiplier
        )
        return particle_belief_state
    
    def full_bayesian_update(
        self, 
        particle_belief_state: ParticleBelief,
        actions: np.array, 
        observations: np.array
    ) -> ParticleBelief:
        # Run dynamics and observation updates
        particle_belief_state = self.dynamics_update(particle_belief_state, actions)
        particle_belief_state = self.observation_update(particle_belief_state, observations)

        # Normalize and resample 
        particle_belief_state.normalize_weights()
        if self.resample:
            particle_belief_state = self._resample_particles(particle_belief_state)
        return particle_belief_state

    def _resample_particles(
        self, 
        particle_belief_state: ParticleBelief,
    ) -> ParticleBelief:
        # Naive resampling - probably not necessary here
        particle_belief_state.normalize_weights()
        sampled_indices = np.choice(
            particle_belief_state.num_particles, 
            particle_belief_state.num_particles, 
            p = particle_belief_state.weights
        )
        particle_belief_state.update_states(
            particle_belief_state.states[sampled_indices]
        )
        particle_belief_state.reinitialize_weights()
        return particle_belief_state

    def sample_new_action(
        self,
        action_set: set
    ) -> int:
        return self.dynamics_and_observation.sample_new_action(action_set)

    def sample_random_state(
        self,
        particle_belief_state: ParticleBelief
    ) -> np.array:
        # Sample a single state from particle belief according to the likelihood weights
        sampled_state_index = np.random.choice(
            particle_belief_state.num_particles, 
            p = particle_belief_state.weights / np.sum(particle_belief_state.weights)
        )
        return particle_belief_state.states[sampled_state_index].reshape([1, -1])

    def generate_next_belief(
        self,
        particle_belief_state: ParticleBelief,
        action: int
    ) -> ParticleBelief:
        # Get a copy of the input
        next_particle_belief_state = deepcopy(particle_belief_state)

        # Sample an observation by first picking a state
        sampled_state = self.sample_random_state(next_particle_belief_state)

        # Then generating the observation from the sampled state (tiled for vectorization)
        sampled_observation = self.dynamics_and_observation.sample_observation(sampled_state).flatten()
        sampled_observation_tiled = np.tile(
            sampled_observation, 
            [next_particle_belief_state.num_particles, 1]
        )
        actions = np.tile(
            action, 
            next_particle_belief_state.num_particles
        )
        
        # Run full bayesian next step with the sampled observation
        next_particle_belief_state = self.full_bayesian_update(
            next_particle_belief_state,
            actions, 
            sampled_observation_tiled
        )

        return next_particle_belief_state