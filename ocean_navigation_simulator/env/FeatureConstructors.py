from ocean_navigation_simulator.env.Arena import ArenaObservation
from ocean_navigation_simulator.env.PlatformState import PlatformState
from ocean_navigation_simulator.env import utils


def DoubleGyreFeatureConstructor(obs: ArenaObservation, target: PlatformState) -> ArenaObservation:
    """
    Converts the observation to use relative positions
    Args:
        obs: current platform observation
        target: goal end state

    Returns:

    """
    relative_obs = utils.euclidean_distance(obs.platform_state, target)

    return ArenaObservation(
        platform_state=relative_obs,
        true_current_at_state=obs.true_current_at_state,
        forecasted_current_at_state=obs.forecasted_current_at_state
    )