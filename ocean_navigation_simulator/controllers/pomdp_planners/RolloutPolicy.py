from typing import Union

import numpy as np


# Heuristic action policy ('naive to target')
class NaiveToTarget:
    """Heuristic policy to go to a target."""
    def __init__(self, target: np.array):
        self.target = target

    def get_action(self, states: np.array) -> np.array:
        """Heuristic policy to go to a target.
        Args:
            states: state vector (n, 5) with columns (x, y, t, period_time, v_amplitude)
        Returns:
            actions: vector of actions as np.array (n,)
        """
        # get the angle to the target
        angle_to_target = np.arctan2(self.target[1] - states[:,1], self.target[0] - states[:, 0])

        # discrete actions (but also negative actions)
        actions = np.round(angle_to_target / (np.pi / 4)).astype(int)

        # now we need to map to only positive actions
        return np.where(actions < 0, actions + 8, actions)