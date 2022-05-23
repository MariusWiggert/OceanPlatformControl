# Make sub-folders visible to the module
from ocean_navigation_simulator import steering_controllers
from ocean_navigation_simulator import planners
from ocean_navigation_simulator import utils
from ocean_navigation_simulator import evaluation
from ocean_navigation_simulator import env


# specify version (for pip installation)
__version__ = "0.0.1"
__all__ = (
    "steering_controllers",
    "planners",
    "evaluation",
    "env",
    "utils",
)