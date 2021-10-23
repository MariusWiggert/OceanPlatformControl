# Make sub-folders visible to the module
from ocean_navigation_simulator import steering_controllers
from ocean_navigation_simulator import planners
from ocean_navigation_simulator import utils
from ocean_navigation_simulator import evaluation

# Make certain classes directly accessible
from ocean_navigation_simulator.simulator import OceanNavSimulator
from ocean_navigation_simulator.problem import Problem, WaypointTrackingProblem

# specify version (for pip installation)
__version__ = "0.3.0"
__all__ = ("OceanNavSimulator", "Problem", "WaypointTrackingProblem",
           "steering_controllers", "utils", "planners", "evaluation")