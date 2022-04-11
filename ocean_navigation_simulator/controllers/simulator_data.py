import dataclasses


@dataclasses.dataclass
class SimulatorObservation(object):
    """
    Specifies an observation from the simulator.
    This differs from SimulatorState in that the observations are not
    ground truth state, and are instead noisy observations from the
    environment.
    """
    platform_observation: platform.PlatformState
    current_at_platform: current_field.CurrentVector
    forecasts: data.Forecast
