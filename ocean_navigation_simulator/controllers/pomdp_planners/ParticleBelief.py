import numpy as np
from typing import Optional

class ParticleBelief:
    def __init__(
        self, 
        states: np.array, 
        weights: Optional[np.array] = None
    ) -> None:
        self.states = states
        self.num_particles = len(self.states)
        self.is_normalized = False
        
        if weights is None or len(weights) == 0:
            self.weights = [1.0] * self.num_particles
        elif len(weights) != self.num_particles:
            raise Exception("Particle Belief Initialization: Numbers of states particles and weights do not match")
        else:
            self.weights = weights
    
    def update_states(self, states: np.array) -> None:
        if states is None:
            raise Exception("Particle Belief State Update: New states is empty")
        if self.num_particles != len(states):
            raise Exception("Particle Belief State Update: New states is not the same length as old states")
        self.states = states

    def update_weights(self, weights: np.array) -> None:
        if self.num_particles != len(weights):
            raise Exception("Particle Belief Weight Update: Numbers of states particles and weights do not match")
        self.weights = weights
        self.is_normalized = False

    def update_belief(self, states: np.array, weights: np.array) -> None:
        self.update_states(states)
        self.update_weights(weights)

    def normalize_weights(self) -> None:
        self.weights /= np.sum(self.weights)
        self.is_normalized = True
    
    def reinitialize_weights(self) -> None:
        self.weights = [1.0] * self.num_particles
        self.normalize_weights()