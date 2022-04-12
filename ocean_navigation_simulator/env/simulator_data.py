import dataclasses

from ocean_navigation_simulator.data_sources import OceanCurrentFields, OceanCurrentVector
from ocean_navigation_simulator.env import platform


@dataclasses.dataclass
class SimulatorObservation(object):
    """
    Specifies an observation from the simulator.
    """
    platform_observation: platform.PlatformState  # Jerome has this
    current_at_platform: OceanCurrentVector.OceanCurrentVector  # Marius has this

    # TODO: currently OceanCurrent Field enables access to both hindcast and forecast;
    #  configure so that observation only includes forecast?
    forecasts: OceanCurrentFields.OceanCurrentField


@dataclasses.dataclass
class SimulatorAction(object):
    """
    thrust -> float, % of maximum thrust
    heading -> float, radians
    """
    thrust: float
    heading: float
