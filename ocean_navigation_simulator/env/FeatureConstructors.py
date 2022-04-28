import numpy as np

from ocean_navigation_simulator.env.Arena import ArenaObservation
from ocean_navigation_simulator.env.DoubleGyreProblem import DoubleGyreProblem

"""
Feature Constructors should take in the ArenaObservation and other relavant information, make any featurization changes,
and then convert to a numpy array that the RL model can use.
"""


def DoubleGyreFeatureConstructor(obs: ArenaObservation, problem: DoubleGyreProblem) -> np.ndarray:
    """
    Converts the observation to use relative positions
    Args:
        problem: class containing information about RL problem (end region, start state, etc.)
        obs: current platform observation

    Returns:

    """
    target = problem.end_region
    lat_diff = target.lat - obs.platform_state.lat
    lon_diff = target.lon - obs.platform_state.lon
    current = obs.true_current_at_state
    time_elapsed = (obs.platform_state.date_time - problem.start_state.date_time).total_seconds()

    return np.array([lat_diff, lon_diff, time_elapsed, current.u, current.v])  # TODO: convert to numpy
